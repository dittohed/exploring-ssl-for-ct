import sys
import math
import argparse
import warnings
import wandb

import torch
import torch.optim as optim

from functools import partial
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.data import (
    DataLoader, 
    ThreadDataLoader, 
    Dataset,
    PersistentDataset, 
    decollate_batch
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_surface_dice
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from monai.utils.enums import MetricReduction

import src.utils as utils

from src.loaders import get_finetune_data
from src.transforms import get_finetune_transforms_2d, get_finetune_transforms_3d
from src.callbacks import EarlyStopping, BestCheckpoint


def get_args_parser():
    parser = argparse.ArgumentParser('Finetune CT')

    # Swin params
    parser.add_argument('--embedding_size', default=24, type=int,
        help='Swin backbone base embedding size (C from the paper).')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
        help='''Whether to use gradient checkpointing (saves memory, 
        longer training, might be useful for 3D).''')
    
    # Data params
    parser.add_argument('--spatial_dims', default=2, type=int, 
        help='Spatial dimension of input data, either 2 for 2D or 3 for 3D.')
    parser.add_argument('--a_min', default=-500, type=float, 
        help='`a_min` in monai.transforms.ScaleIntensityRanged.')
    parser.add_argument('--a_max', default=500, type=float, 
        help='`a_max` in monai.transforms.ScaleIntensityRanged.')
    parser.add_argument('--size_x', default=1.0, type=float, 
        help='Pixel size in x direction.')
    parser.add_argument('--size_y', default=1.0, type=float, 
        help='Pixel size in y direction.')
    parser.add_argument('--size_z', default=2.5, type=float, 
        help='Pixel size in z direction.')
    parser.add_argument('--n_classes', default=14, type=int,
        help='Number of segmentation classes (= number of output channels).')

    # Training params
    parser.add_argument('--use_amp', action='store_true',
        help='Whether to use Automatic Mixed Precision for training.')
    parser.add_argument('--batch_size', default=2, type=int,
        help='No. of unique CT images in minibatch (see also --n_crops).')
    parser.add_argument('--n_crops_per_ct', default=2, type=int,
        help='No. of crops returned for each CT image in minibatch.')
    parser.add_argument('--sw_batch_size', default=4, type=int,
        help='Batch size for sliding window inference.')
    parser.add_argument('--n_epochs', default=500, type=int, 
        help='Number of epochs of training.')
    parser.add_argument('--base_lr', default=1e-3, type=float, 
        help='''Learning rate at the end of linear warmup (highest used during 
        training).''')
    parser.add_argument('--warmup_epochs', default=10, type=int,
        help='Number of epochs for the linear learning-rate warm up.')
    parser.add_argument('--wd', type=float, default=1e-5, 
        help='Weight decay throughout the whole training.')
    parser.add_argument('--sw_overlap', default=0.25, type=float,
        help='Sliding window inference overlap.')
    parser.add_argument('--patience', default=10, type=float,
        help='How many evals to wait for val metric to improve before terminating.')

    # Other params
    parser.add_argument('--run_name', default='test_finetune', type=str,
        help='Unique run/experiment name.')
    parser.add_argument('--eval_every', default=10, type=int,
        help='After how many epochs to evaluate in the training cycle.')
    parser.add_argument('--eval_train', action='store_true',
        help='Whether to evaluate also using training data besides validation data.')
    parser.add_argument('--data_dir', default='./data/finetune_preprocessed_2d', type=str,
        help='Path to training data directory.')
    parser.add_argument('--split_path', default='./data/split.json', type=str,
        help='Path to .json file with data split.')
    parser.add_argument('--chkpt_dir', default='./chkpts', type=str, 
        help='Path to directory for storing trained model\'s best checkpoint.')
    parser.add_argument('--chkpt_path', type=str, 
        help='''Path to model checkpoint to load at the beginning of training.
        If not provided, the model will be trained from scratch.''')
    parser.add_argument('--cache_dir', default='./cache', type=str, 
        help='`cache_dir` in monai.data.PersistentDataset objects.')
    parser.add_argument('--seed', default=4294967295, type=int, 
        help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, 
        help='Number of data loading workers, used only if --spatial_dims 2.')
    parser.add_argument('--use_wandb', action='store_true',
        help='Whether to log training config and results to W&B.')
    parser.add_argument('--low_resource_mode', action='store_true',
        help='Whether to limit memory footprint for minor tests.')
    parser.add_argument('--ignore_user_warning', action='store_true',
        help='''Whether to ignore UserWarning raised by 
        `monai.transforms.RandCropByPosNegLabeld`.''')

    return parser


def train_one_epoch(model, loss_fn, train_loader, optimizer, lr_schedule, 
        epoch, scaler, args, device):
    avg_loss = utils.AverageAggregator()
    tqdm_it = tqdm(train_loader, total=len(train_loader), leave=True)
    tqdm_it.set_description(f'Epoch: [{epoch+1}/{args.n_epochs}]')

    for batch_idx, data_dict in enumerate(tqdm_it):
        # Prepare input
        img, label = data_dict['img'].to(device), data_dict['label'].to(device)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            pred = model(img)
            loss = loss_fn(pred, label)

        if not math.isfinite(loss.item()):
            print(f'Loss is {loss.item()}, stopping training...')
            sys.exit(1)

        # utils.display_gpu_info()

        # Optimize
        step = len(train_loader) * epoch + batch_idx  # Calculate global step number
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[step]

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()

        # Logging
        tqdm_it.set_postfix(
            loss=str(loss.item()), 
            lr=lr_schedule[step]
        )
        avg_loss.update(loss.item())

    log_dict = {
        'train/loss': avg_loss.item(),
        'train/lr': lr_schedule[step]
    }
    return log_dict


@torch.no_grad()
def evaluate(subset, model, acc_fn, data_loader, post_label, post_pred,
        scaler, org_thresholds, args, device):
    avg_dice = utils.AverageAggregator()
    avg_surf_dice = utils.AverageAggregator()

    for data_dict in data_loader:
        img, label = data_dict['img'].to(device), data_dict['label'].to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            pred = model(img)

        label_list = decollate_batch(label)
        label_list = [post_label(label_tensor) for label_tensor in label_list]
        pred_list = decollate_batch(pred)
        pred_list = [post_pred(pred_tensor) for pred_tensor in pred_list]

        # Dice
        acc_fn.reset()
        acc_fn(y_pred=pred_list, y=label_list)
        acc, not_nans = acc_fn.aggregate()
        assert not_nans == 1  # TODO: be careful for multiple GPUs
        avg_dice.update(acc.item())

        # Surface dice
        surf_dice = compute_surface_dice(
            y_pred=torch.stack(pred_list), 
            y=torch.stack(label_list), 
            class_thresholds=list(org_thresholds.values()),
            spacing=(args.size_y, args.size_x, args.size_z)[:args.spatial_dims]
        )
        # torch.nanmean() to ignore cases where there's no certain class
        # neither in pred nor in gt 
        avg_surf_dice.update(torch.nanmean(surf_dice))  

    print(f'Mean {subset} dice score: {avg_dice.item():.4f}.')
    print(f'Mean {subset} surface dice score: {avg_surf_dice.item():.4f}.')

    log_dict = {
        f'{subset}/dice': avg_dice.item(),
        f'{subset}/surf_dice': avg_surf_dice.item()
    }
    return log_dict


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_determinism(args.seed)
    torch.backends.cudnn.benchmark = True

    # Prepare data
    train_data, val_data = get_finetune_data(
        Path(args.data_dir),
        Path(args.split_path)
    )

    if args.spatial_dims == 3:
        train_transforms, val_transforms = get_finetune_transforms_3d(args, device)
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        train_ds = PersistentDataset(
            data=train_data, 
            transform=train_transforms,
            cache_dir=args.cache_dir
        )
        val_ds = PersistentDataset(
            data=val_data, 
            transform=val_transforms,
            cache_dir=args.cache_dir
        )
        train_loader = ThreadDataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            num_workers=0,
            shuffle=True
        )
        train_eval_loader = ThreadDataLoader(
            train_ds, 
            batch_size=1,
            num_workers=0, 
            shuffle=False
        )
        val_loader = ThreadDataLoader(
            val_ds, 
            batch_size=1,
            num_workers=0, 
            shuffle=False
        )
    else:
        train_transforms, val_transforms = get_finetune_transforms_2d(args)
        train_ds = Dataset(
            data=train_data,
            transform=train_transforms
        )
        val_ds = Dataset(
            data=val_data, 
            transform=val_transforms
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )

    # Prepare model
    model = SwinUNETR(
        img_size=tuple([96]*args.spatial_dims),
        in_channels=1,
        out_channels=args.n_classes,
        feature_size=args.embedding_size,
        num_heads=(3, 3, 3, 3) if args.low_resource_mode else (3, 6, 12, 24),
        spatial_dims=args.spatial_dims,
        use_checkpoint=args.use_gradient_checkpointing
    ).to(device)

    if args.chkpt_path is not None:
        model.swinViT.load_state_dict(
            torch.load(args.chkpt_path)
        )
        print(f'Successfully loaded weights from {args.chkpt_path}.')

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
        end_val=5e-5,
        n_epochs=args.n_epochs,
        iters_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs
    )

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Prepare other stuff for validation
    acc_fn = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    post_label = AsDiscrete(to_onehot=args.n_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.n_classes)

    org_thresholds = OrderedDict(  # FLARE2022 official thresholds
        {'Liver': 5, 'RK': 3, 'Spleen': 3, 'Pancreas': 5, 
         'Aorta': 2, 'IVC': 2, 'RAG': 2, 'LAG': 2, 'Gallbladder': 2,
         'Esophagus': 3, 'Stomach': 5, 'Duodenum': 7, 'LK': 3})

    model_infer = partial(
        sliding_window_inference,
        roi_size=tuple([96]*args.spatial_dims),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.sw_overlap
    )

    bc = BestCheckpoint(
        model=model, 
        save_path=Path(args.chkpt_dir)/Path(args.run_name+'_best.pt')
    )
    es = EarlyStopping(args.patience)

    if args.ignore_user_warning:
        warnings.filterwarnings(
            action='ignore',
            message='.*unable to generate class balanced samples.*',
        )

    # Train
    for epoch in range(args.n_epochs):

        model.train()
        log_dict = train_one_epoch(
            model, loss_fn, train_loader, optimizer, lr_schedule, 
            epoch, scaler, args, device
        )

        if (epoch+1) % args.eval_every == 0:
            model.eval()

            if args.eval_train:
                eval_log_dict_train = evaluate(
                    'train', model_infer, acc_fn, train_eval_loader, post_label, 
                    post_pred, scaler, org_thresholds, args, device
                )
                log_dict.update(eval_log_dict_train)

            eval_log_dict_val = evaluate(
                'val', model_infer, acc_fn, val_loader, post_label, 
                post_pred, scaler, org_thresholds, args, device
            )
            log_dict.update(eval_log_dict_val)
            bc(eval_log_dict_val['val/dice'])
            es(eval_log_dict_val['val/dice'])

        if args.use_wandb:
            wandb.log(log_dict)

        if es.terminate:
            break


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.low_resource_mode:
        args.eval_every = 1
        args.embedding_size = 12
        args.batch_size = 1
        args.n_crops_per_ct = 2
        args.sw_batch_size = 1
        args.sw_overlap = 0

    if args.use_wandb:
        wandb.init(
            project='exploring-ssl-for-ct-tune',
            name=args.run_name,
            config=vars(args)
        )
        wandb.define_metric('train/loss', summary='min')
        wandb.define_metric('val/dice', summary='max')

    main(args)