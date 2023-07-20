import argparse
import wandb

from pathlib import Path

import torch

from tqdm import tqdm
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from lightly.loss import NegativeCosineSimilarity

import src.utils as utils

from src.loaders import get_ssl_data
from src.transforms import get_ssl_transforms_2d, get_ssl_transforms_3d
from src.models import Backbone, SimSiam


def get_args_parser():
    parser = argparse.ArgumentParser('Pretrain CT using SimSiam')

    # Swin params
    parser.add_argument('--embedding_size', default=24, type=int,
        help='Swin backbone base embedding size (C from the paper).')
    parser.add_argument('--drop_path_rate', default=0.1, type=float,
        help='`drop_path_rate` for monai.networks.nets.swin_unetr.SwinTransformer.')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
        help='Whether to use gradient checkpointing (saves memory, longer training).')
    
    # Data params
    parser.add_argument('--spatial_dims', default=2, type=int, 
        help='Spatial dimension of input data, either 2 for 2D or 3 for 3D')
    parser.add_argument('--a_min', default=-500, type=float, 
        help='`a_min` in monai.transforms.ScaleIntensityRanged')
    parser.add_argument('--a_max', default=500, type=float, 
        help='`a_max` in monai.transforms.ScaleIntensityRanged')
    parser.add_argument('--size_x', default=1.0, type=float, 
        help='Pixel size in x direction')
    parser.add_argument('--size_y', default=1.0, type=float, 
        help='Pixel size in y direction')
    parser.add_argument('--size_z', default=2.5, type=float, 
        help='Pixel size in z direction')
    parser.add_argument('--min_iou', default=0, type=float, 
        help='Min. IoU of the 2nd crop with the 1st crop')
    parser.add_argument('--max_iou', default=1.0, type=float, 
        help='Max. IoU of the 2nd crop with the 1st crop')

    # Training params
    parser.add_argument('--use_amp', action='store_true',
        help='Whether to use Automatic Mixed Precision for training.')
    parser.add_argument('--batch_size', default=2, type=int,
        help='''Number of distinct images for which a single
        backward pass will be calculated (just a batch size if running with
        --accum_iters 1).''')
    parser.add_argument('--n_epochs', default=300, type=int, 
        help='Number of epochs of training.')
    parser.add_argument('--base_lr', default=0.5, type=float,
        help='''Learning rate at the end of linear warmup (highest used during 
        training).''')
    parser.add_argument('--warmup_epochs', default=0, type=int,
        help='Number of epochs for the linear learning-rate warm up.')
    parser.add_argument('--end_lr', type=float, default=1e-6,
        help='''Target lr at the end of optimization. We use a cosine lr 
        schedule with linear warmup.''')
    parser.add_argument('--wd', type=float, default=1e-5, 
        help='Weight decay throughout the training.')
    parser.add_argument('--accum_iters', type=int, default=1,
        help='How many backward passes to calculate before calling optimizer.step().')

    # Other params
    parser.add_argument('--run_name', default='test_ssl', type=str,
        help='Unique run/experiment name.')
    parser.add_argument('--data_dir', default='./data/ssl_preprocessed_2d', type=str,
        help='Path to pretraining data directory.')
    parser.add_argument('--chkpt_dir', default='./chkpts', type=str, 
        help='Path to directory for storing trained model\'s last checkpoint.')
    parser.add_argument('--seed', default=4294967295, type=int, 
        help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, 
        help='''Number of data loading workers. Should remain
        0 for --spatial_dims 3 as GPU is used to perform transformations (faster
        but can't be parallelized using `DataLoader`).
        If -1, runs quick benchmark first to pick the best value.''')
    parser.add_argument('--use_wandb', action='store_true',
        help='Whether to log training config and results to W&B.')
    parser.add_argument('--low_resource_mode', action='store_true',
        help='Whether to limit memory footprint for minor tests.')

    return parser


def train_one_epoch(model, loss_fn, train_loader, iters_per_epoch,
        optimizer, lr_schedule, epoch, scaler, device, args):
    avg_loss = utils.AverageAggregator()
    batch_loss = 0  # Accumulate loss from accumulation steps

    # Display tqdm for each backward pass (actual batches)
    # Update metrics only after optimizer.step() call
    tqdm_it = tqdm(train_loader, total=iters_per_epoch*args.accum_iters, leave=True)
    tqdm_it.set_description(f'Epoch: [{epoch+1}/{args.n_epochs}]')

    for batch_idx, data_dict in enumerate(tqdm_it):
        # Check logical batch number and skip for last incomplete batch
        if batch_idx // args.accum_iters == iters_per_epoch:
            break

        x1, x2 = data_dict['img1'].to(device), data_dict['img2'].to(device)
    
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # Forward pass
            z1, p1 = model(x1)
            z2, p2 = model(x2)

            loss = 0.5 * (loss_fn(z1, p2) + loss_fn(z2, p1))
            loss = loss / args.accum_iters

        # utils.display_gpu_info()
        batch_loss += loss.item()

        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # If next batch belongs to a new logical batch
        # i.e. this is the last batch to accumulate
        if (batch_idx+1) % args.accum_iters == 0:
            # Calculate global logical batch number
            step = iters_per_epoch * epoch + batch_idx // args.accum_iters

            for i, param_group in enumerate(optimizer.param_groups):
                if i < 4:  # No lr decay for prediction head
                    param_group['lr'] = lr_schedule[step]

            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Logging
            tqdm_it.set_postfix(  
                loss=str(batch_loss),  # str() for no rounding
                lr=lr_schedule[step]
            )  
            avg_loss.update(batch_loss)
            
            # Starting new accumulation
            batch_loss = 0

    log_dict = {
        'train/loss': avg_loss.item(),
        'train/lr': lr_schedule[step]
    }
    return log_dict


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_determinism(args.seed)

    # Prepare data
    if args.spatial_dims == 2:
        transforms = get_ssl_transforms_2d(args)
    else:
        transforms = get_ssl_transforms_3d(args, device=device)

    ds = Dataset(
        data=get_ssl_data(args.data_dir), 
        transform=transforms
    )

    if args.num_workers == -1:
        num_workers = utils.get_best_workers(ds, args.batch_size)
    else:
        num_workers = args.num_workers
    data_loader = DataLoader(
        ds, 
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    # Prepare model
    model = SimSiam(
        backbone=Backbone(args),
        backbone_out_dim=args.embedding_size*2**4
    ).to(device)

    # Prepare other stuff for training
    loss_fn = NegativeCosineSimilarity()

    # Don't regularize biases and norm layers
    # Don't apply lr schedule to prediction head (handled in loop)
    param_groups_backbone = utils.get_param_groups(model.backbone)
    param_groups_projection = utils.get_param_groups(model.projection_head)
    param_groups_prediction = utils.get_param_groups(model.prediction_head)

    param_groups = (
        param_groups_backbone + param_groups_projection + param_groups_prediction
    )
    optimizer = torch.optim.SGD(
        param_groups, 
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd
    )

    # Specify the number of optimizer.step() calls (logical batches)
    # This is needed for turning gradient accum on/off smoothly
    # Last incomplete logical batch is skipped
    iters_per_epoch = len(ds) // (args.batch_size*args.accum_iters)

    lr_schedule = utils.cosine_scheduler(
        base_val=args.base_lr,
        end_val=args.end_lr,
        n_epochs=args.n_epochs,
        iters_per_epoch=iters_per_epoch,
        warmup_epochs=args.warmup_epochs
    )
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    Path(args.chkpt_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for epoch in range(args.n_epochs):
        log_dict = train_one_epoch(
            model, loss_fn, data_loader, iters_per_epoch, optimizer,
            lr_schedule, epoch, scaler, device, args
        )

        torch.save(
            model.state_dict(), 
            Path(args.chkpt_dir)/Path(args.run_name+'.pt')
        )

        if args.use_wandb:
            wandb.log(log_dict)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.low_resource_mode:
        args.embedding_size = 12
        args.batch_size = 4

    if args.use_wandb:
        wandb.init(
            project='exploring-ssl-for-ct-pre',
            name=args.run_name,
            config=vars(args)
        )
        wandb.define_metric('train/loss', summary='min')

    # with open(f'{args.run_name}_args.json', 'w') as outfile:
    #     json.dump(vars(args), outfile)
    
    main(args)