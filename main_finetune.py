import sys
import math
import argparse
import wandb

import torch
import torch.optim as optim

from functools import partial
from pathlib import Path
from tqdm import tqdm

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.data import ThreadDataLoader, CacheDataset, set_track_meta, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from monai.utils.enums import MetricReduction

import src.utils as utils

from src.loaders import get_finetune_data
from src.transforms import get_finetune_transforms
from src.callbacks import EarlyStopping, BestCheckpoint


def get_args_parser():
    parser = argparse.ArgumentParser('Finetune CT')  # TODO: add_help?

    # Swin params
    parser.add_argument('--embedding_size', default=12, type=int,  # TODO: back to 48
        help='Swin backbone base embedding size (C from the paper).')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',  # TODO: could try
        help='Whether to use gradient checkpointing (saves memory, longer training).')
    
    # Data params
    parser.add_argument('--spatial_dims', default=3, type=int, 
        help='Spatial dimension of input data, either 2 for 2D or 3 for 3D.')
    parser.add_argument('--a_min', default=-175, type=float, 
        help='`a_min` in monai.transforms.ScaleIntensityRanged.')
    parser.add_argument('--a_max', default=250, type=float, 
        help='`a_max` in monai.transforms.ScaleIntensityRanged.')
    parser.add_argument('--size_x', default=1.5, type=float, 
        help='Pixel size in x direction.')
    parser.add_argument('--size_y', default=1.5, type=float, 
        help='Pixel size in y direction.')
    parser.add_argument('--size_z', default=2.0, type=float, 
        help='Pixel size in z direction.')
    parser.add_argument('--cache_num', default=4, type=float,  # TODO: back to 24
        help='`cache_num` in monai.data.CacheDataset.')
    parser.add_argument('--n_classes', default=14, type=int,
        help='Number of segmentation classes (= number of output channels).')

    # Training params
    parser.add_argument('--use_fp16', action='store_true',
        help='''Whether or not to use half precision for training. 
        Improves training time and memory requirements, but can provoke instability 
        and slight decay of performance. We recommend disabling mixed precision if 
        the loss is unstable, if reducing the patch size or if training with bigger ViTs.''')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,  # TODO: back to 2
        help='`num_samples` in monai.transforms.RandCropByPosNegLabeld (per GPU).')
    parser.add_argument('--sw_batch_size', default=2, type=int,  # TODO: back to 4
        help='Batch size for sliding window inference.')
    parser.add_argument('--n_epochs', default=5000, type=int, 
        help='Number of epochs of training.')
    parser.add_argument('--base_lr', default=1e-4, type=float, 
        help='''Learning rate at the end of linear warmup (highest used during 
        training).''')
    parser.add_argument('--warmup_epochs', default=50, type=int,
        help='Number of epochs for the linear learning-rate warm up.')
    parser.add_argument('--wd', type=float, default=1e-5, 
        help='Weight decay throughout the whole training.')
    parser.add_argument('--sw_overlap', default=0.2, type=float,
        help='Sliding window inference overlap.')  # TODO: 0.5 might give better results
    parser.add_argument('--patience', default=10, type=float,
        help='How many epochs to wait for val metric to improve before terminating.')

    # Other params
    parser.add_argument('--run_name', default='test', type=str,
        help='Unique run/experiment name.')
    parser.add_argument('--eval_every', default=1, type=int,
        help='After how many epochs to evaluate.')
    parser.add_argument('--data_dir', default='./data/finetune', type=str,
        help='Path to training data directory.')
    parser.add_argument('--chkpt_dir', default='./chkpts', type=str, 
        help='Path to save checkpoints.')
    parser.add_argument('--seed', default=4294967295, type=int, 
        help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, 
        help='Number of data loading workers per GPU.')
    parser.add_argument('--use_wandb', action='store_true',
        help='Whether to log training config and results to W&B.')

    return parser


def train_one_epoch(model, loss_fn, train_loader, optimizer, lr_schedule, 
        epoch, fp16_scaler, device, args):
    tqdm_it = tqdm(train_loader, total=len(train_loader), leave=True)
    tqdm_it.set_description(f'Epoch: [{epoch+1}/{args.n_epochs}]')

    for batch_idx, data_dict in enumerate(tqdm_it):
        # Prepare input
        img, label = data_dict['img'].to(device), data_dict['label'].to(device)

        # Forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            pred = model(img)
            loss = loss_fn(pred, label)

        if not math.isfinite(loss.item()):
            print(f'Loss is {loss.item()}, stopping training...')
            sys.exit(1)

        # utils.display_gpu_info()
        
        # Backward pass
        loss.backward()

        # Optimize
        step = len(train_loader) * epoch + batch_idx  # Calculate global step number
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[step]

        optimizer.step()
        optimizer.zero_grad()

        # Logging
        tqdm_it.set_postfix(
            loss=str(loss.item()), 
            lr=lr_schedule[step]
        )

        if args.use_wandb:
            wandb.log({
                'loss': loss.item(),
                'lr': lr_schedule[step]
            })


@torch.no_grad()
def val_one_epoch(model, acc_fn, val_loader, post_label, post_pred,
        fp16_scaler, device):
    avg_agg = utils.AverageAggregator()

    for data_dict in val_loader:
        img, label = data_dict['img'].to(device), data_dict['label'].to(device)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            pred = model(img)

        label_list = decollate_batch(label)
        label_list = [post_label(label_tensor) for label_tensor in label_list]
        pred_list = decollate_batch(pred)
        pred_list = [post_pred(pred_tensor) for pred_tensor in pred_list]

        acc_fn.reset()
        acc_fn(y_pred=pred_list, y=label_list)
        acc, not_nans = acc_fn.aggregate()
        assert not_nans == 1  # TODO: be careful for multiple GPUs
        avg_agg.update(acc.item())

    print(f'Mean validation dice score: {avg_agg.item():.4f}')
    if args.use_wandb:
        wandb.log({
            'val_dice': avg_agg.item()
        })

    return avg_agg.item()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_determinism(args.seed)
    # set_track_meta(False)
    # TODO: see if this will still cause problems if using for uncached samples
    # and moved after train_ds and val_ds construction

    # Prepare data
    train_data, val_data = get_finetune_data(args.data_dir)
    train_transforms, val_transforms = get_finetune_transforms(args)

    train_ds = CacheDataset(
        data=train_data, 
        transform=train_transforms,
        cache_num=args.cache_num,
        num_workers=8  # TODO: check optimal
    )
    train_loader = ThreadDataLoader(
        train_ds, 
        num_workers=0,
        batch_size=1, 
        shuffle=True
    )

    val_ds = CacheDataset(
        data=val_data, 
        transform=val_transforms,
        cache_num=args.cache_num//2,
        num_workers=8//2  # TODO: check optimal
    )
    val_loader = ThreadDataLoader(
        val_ds, 
        num_workers=0, 
        batch_size=1, 
        shuffle=False
    )

    # Prepare model
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=args.n_classes,
        feature_size=args.embedding_size,
        num_heads=(3, 3, 3, 3),  # TODO: [3, 6, 12, 24] originally
        use_checkpoint=args.use_gradient_checkpointing
    ).to(device)

    # TODO: load from finetune checkpoint or ssl checkpoint

    # Prepare other stuff for training
    loss_fn = DiceCELoss(
        to_onehot_y=True, 
        softmax=True
    )

    param_groups = utils.get_param_groups(model)
    param_groups[0]['weight_decay'] = args.wd
    optimizer = optim.AdamW(params=param_groups)

    lr_schedule = utils.cosine_scheduler(
        base_val=args.base_lr,
        end_val=0,
        n_epochs=args.n_epochs,
        iters_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs
    )

    # Prepare other stuff for validation
    acc_fn = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    post_label = AsDiscrete(to_onehot=args.n_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.n_classes)

    model_infer = partial(
        sliding_window_inference,
        roi_size=[96, 96, 96],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.sw_overlap
    )

    bc = BestCheckpoint(
        model=model, 
        save_path=Path(args.chkpt_dir)/Path(args.run_name+'.pt')
    )
    es = EarlyStopping(args.patience)

    # Train
    for epoch in range(args.n_epochs):

        model.train()
        train_one_epoch(
            model, loss_fn, train_loader, optimizer, lr_schedule, 
            epoch, None, device, args)

        if epoch % args.eval_every == 0:
            model.eval()
            val_score = val_one_epoch(model_infer, acc_fn, val_loader, 
                                      post_label, post_pred,
                                      None, device)
            bc(val_score)
            if es(val_score):
                break


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.use_wandb:
        wandb.login()
        wandb.init(
            project='exploring-ssl-for-ct',
            name=args.run_name,
            config=vars(args)
        )

    main(args)

    if args.use_wandb:
        wandb.finish()
        