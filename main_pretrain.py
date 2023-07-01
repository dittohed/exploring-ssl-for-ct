import sys
import math
import argparse
import wandb

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism

import src.utils as utils

from src.loaders import get_ssl_data
from src.transforms import get_ssl_transforms
from src.models import Backbone
from src.dino import Head as DINOHead
from src.dino import Loss as DINOLoss


def get_args_parser():
    parser = argparse.ArgumentParser('Pretrain CT')

    # Swin params
    parser.add_argument('--embedding_size', default=48, type=int,
        help='Swin backbone base embedding size (C from the paper).')
    parser.add_argument('--drop_path_rate', default=0.1, type=float,
        help='`drop_path_rate` for monai.networks.nets.swin_unetr.SwinTransformer.')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
        help='Whether to use gradient checkpointing (saves memory, longer training).')
    
    # DINO head params
    parser.add_argument('--out_dim', default=1024, type=int,
        help='Dimensionality of the last head layer (softmax is calculated on).')
    
    # DINO loss params
    parser.add_argument('--init_teacher_temp', default=0.04, type=float,
        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
        help='Final value (after linear warmup) of the teacher temperature.')
    parser.add_argument('--teacher_temp_warmup_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature.')
    
    # Data params
    parser.add_argument('--spatial_dims', default=3, type=int, 
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
    parser.add_argument('--preprocess_mode', default='full', type=str, 
        help='How to preprocess training data.')

    # Training params
    parser.add_argument('--use_amp', action='store_true',
        help='Whether to use AMP for training.')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
        help='''Number of distinct images loaded on a single GPU for which a single
        backward pass will be calculated (just a batch size per GPU if calling with
        --accum_iters 1).''')
    parser.add_argument('--n_epochs', default=300, type=int, 
        help='Number of epochs of training.')
    parser.add_argument('--base_lr', default=5e-5, type=float,
        help='''Learning rate at the end of linear warmup (highest used during 
        training).''')
    parser.add_argument('--warmup_epochs', default=10, type=int,
        help='Number of epochs for the linear learning-rate warm up.')
    parser.add_argument('--end_lr', type=float, default=1e-6,
        help='''Target lr at the end of optimization. We use a cosine lr 
        schedule with linear warmup.''')
    parser.add_argument('--base_wd', type=float, default=0.04, 
        help='Weight decay at the beginning of training.')
    parser.add_argument('--end_wd', type=float, default=0.4, 
        help='Weight decay at the end of training (cosine schedule).')
    parser.add_argument('--base_momentum', type=float, default=0.9995,
        help='Lambda for momentum teacher update.')
    parser.add_argument('--accum_iters', type=int, default=1,
        help='How many backward passes to calculate before calling optimizer.step().')
    parser.add_argument('--freeze_last_layer', default=1, type=int, 
        help='''Number of epochs during which output layer is kept fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.''')
    parser.add_argument('--clip_grad', type=float, default=3.0, 
        help='Maximal parameter gradient norm if using gradient clipping.')

    # Other params
    parser.add_argument('--run_name', default='test_ssl', type=str,
        help='Unique run/experiment name.')
    parser.add_argument('--data_dir', default='./data/ssl', type=str,
        help='Path to pretraining data directory.')
    parser.add_argument('--chkpt_dir', default='./chkpts', type=str, 
        help='Path to checkpoints directory.')
    parser.add_argument('--seed', default=4294967295, type=int, 
        help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, 
        help='Number of data loading workers per GPU.')
    parser.add_argument('--use_wandb', action='store_true',
        help='Whether to log training config and results to W&B.')
    parser.add_argument('--low_resource_mode', action='store_true',
        help='Whether to limit memory footprint for minor tests.')

    return parser


def train_one_epoch(student, teacher, loss_fn, train_loader, iters_per_epoch,
        optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
        scaler, device, args):
    avg_loss = utils.AverageAggregator()
    batch_loss = 0  # Accumulate loss from accumulation steps
    batch_center = torch.zeros_like(loss_fn.center)  # Accumulate batch center

    # Display tqdm for each backward pass (actual batches)
    # Update metrics only after optimizer.step() call
    tqdm_it = tqdm(train_loader, total=iters_per_epoch*args.accum_iters, leave=True)
    tqdm_it.set_description(f'Epoch: [{epoch+1}/{args.n_epochs}]')

    for batch_idx, data_dict in enumerate(tqdm_it):
        # Check logical batch number and skip for last incomplete batch
        if batch_idx // args.accum_iters == iters_per_epoch:
            break

        # Prepare input
        x1, x2 = data_dict['img1'], data_dict['img2']

        if args.low_resource_mode:
            x_student = x1.to(device)
            x_teacher = x2.to(device)
        else:
            # Concat to calculate the loss symmetrically
            x_student = torch.cat([x1, x2]).to(device)
            x_teacher = torch.cat([x2, x1]).to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # Forward pass
            out_student = student(x_student)
            out_teacher = teacher(x_teacher)

            with torch.no_grad():
                batch_center += (
                    torch.mean(out_teacher, dim=0, keepdim=True) / args.accum_iters
                )

            loss = loss_fn(out_student, out_teacher, epoch)
            loss = loss / args.accum_iters

        if not math.isfinite(loss.item()):
            print(f'Loss is {loss.item()}, stopping training...')
            sys.exit(1)

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
                param_group['lr'] = lr_schedule[step]
                if i == 0:  # Only the first group is regularized
                    param_group['weight_decay'] = wd_schedule[step]

            utils.cancel_gradients_last_layer(
                epoch, student, args.freeze_last_layer)

            if args.use_amp:
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                    utils.clip_gradients(student, args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad:
                    utils.clip_gradients(student, args.clip_grad)
                optimizer.step()

            optimizer.zero_grad()

            # Update teacher weights using EMA
            with torch.no_grad():
                m = momentum_schedule[step]
                for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
                    param_teacher.mul_(m).add_(
                        (1-m) * param_student.detach()
                    )

            # Logging
            tqdm_it.set_postfix(  
                loss=str(batch_loss),  # str() for no rounding
                lr=lr_schedule[step],
                wd=wd_schedule[step],
                momentum=str(momentum_schedule[step])
            )  
            avg_loss.update(batch_loss)
            
            # Starting new accumulation
            batch_loss = 0
            loss_fn.update_center(batch_center)
            batch_center = torch.zeros_like(loss_fn.center)

    log_dict = {
        'train/loss': avg_loss.item(),
        'train/lr': lr_schedule[step],
        'train/wd': wd_schedule[step],
        'train/momentum': momentum_schedule[step]
    }
    return log_dict


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_determinism(args.seed)

    # Prepare data
    dataset = Dataset(
        data=get_ssl_data(args.data_dir), 
        transform=get_ssl_transforms(args, args.preprocess_mode, device))
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers
    )

    # Prepare models
    student = nn.Sequential(
        Backbone(args), 
        DINOHead(
            in_dim=args.embedding_size*2**4,
            out_dim=args.out_dim,
        )
    ).to(device)
    teacher = nn.Sequential(
        Backbone(args), 
        DINOHead(
            in_dim=args.embedding_size*2**4,
            out_dim=args.out_dim,
        )
    ).to(device)

    # Student and teacher start with the same weights
    teacher.load_state_dict(student.state_dict())

    # Teacher won't use backprop anyway
    for p in teacher.parameters():
        p.requires_grad = False

    # Prepare other stuff for training
    loss_fn = DINOLoss(
        out_dim=args.out_dim, 
        temp_t_warmup=args.init_teacher_temp,
        temp_t=args.teacher_temp,
        temp_t_warmup_epochs=args.teacher_temp_warmup_epochs,
        n_epochs=args.n_epochs
    ).to(device)

    param_groups = utils.get_param_groups(student)
    optimizer = optim.AdamW(params=param_groups)

    # Specify the number of optimizer.step() calls (logical batches)
    # This is needed for turning gradient accum on/off smoothly
    # Last incomplete logical batch is skipped
    iters_per_epoch = len(dataset) // (args.batch_size_per_gpu*args.accum_iters)

    lr_schedule = utils.cosine_scheduler(
        base_val=args.base_lr,
        end_val=args.end_lr,
        n_epochs=args.n_epochs,
        iters_per_epoch=iters_per_epoch,
        warmup_epochs=args.warmup_epochs
    )
    wd_schedule = utils.cosine_scheduler(
        base_val=args.base_wd,
        end_val=args.end_wd,
        n_epochs=args.n_epochs,
        iters_per_epoch=iters_per_epoch
    )
    momentum_schedule = utils.cosine_scheduler(
        base_val=args.base_momentum,
        end_val=1,
        n_epochs=args.n_epochs,
        iters_per_epoch=iters_per_epoch
    )

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    Path(args.chkpt_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for epoch in range(args.n_epochs):
        log_dict = train_one_epoch(
            student, teacher, loss_fn, data_loader, iters_per_epoch, optimizer,
            lr_schedule, wd_schedule, momentum_schedule, epoch, scaler, device,
            args
        )

        torch.save(
            student[0].model.state_dict(), 
            Path(args.chkpt_dir)/Path(args.run_name+'.pt')
        )

        if args.use_wandb:
            wandb.log(log_dict)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.low_resource_mode:
        args.embedding_size = 12
        args.batch_size_per_gpu = 1

    if args.use_wandb:
        wandb.init(
            project='exploring-ssl-for-ct-pre',
            name=args.run_name,
            config=vars(args)
        )
        wandb.define_metric('train/loss', summary='min')

    main(args)