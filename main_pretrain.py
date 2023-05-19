import sys
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from monai.data import DataLoader, Dataset

import src.utils as utils

from src.loaders import get_ssl_data
from src.transforms import get_ssl_transforms
from src.models import Backbone
from src.dino import Head as DINOHead
from src.dino import Loss as DINOLoss


def get_args_parser():
    parser = argparse.ArgumentParser('Pretrain CT')  # TODO: add_help?

    # Swin params - OK
    parser.add_argument('--embedding_size', default=48, type=int,
        help='Swin backbone base embedding size (C from the paper)')
    parser.add_argument('--dropout_path_rate', default=0.0, type=float,  # TODO: DINO used 0.1 for ViT
        help='TODO')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',  # TODO: could try
        help='Whether to use gradient checkpointing (saves memory, longer training).')
    
    # DINO head params - OK
    parser.add_argument('--out_dim', default=512, type=int,  # TODO: get back to 2048 or as big as possible
        help='Dimensionality of the last head layer (softmax is calculated on).')
    
    # DINO loss params - OK
    parser.add_argument('--init_teacher_temp', default=0.04, type=float,
        help='''Initial value for the teacher temperature: 0.04 works well in most 
        cases. Try decreasing it if the training loss does not decrease.''')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
        help='''Final value (after linear warmup) of the teacher temperature. 
        For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if 
        needed.''')
    parser.add_argument('--teacher_temp_warmup_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature.')
    
    # Data params - OK
    parser.add_argument('--spatial_dims', default=3, type=int, 
        help='Spatial dimension of input data, either 2 for 2D or 3 for 3D')
    parser.add_argument('--a_min', default=-1000, type=float, 
        help='`a_min` in monai.transforms.ScaleIntensityRanged')
    parser.add_argument('--a_max', default=1000, type=float, 
        help='`a_max` in monai.transforms.ScaleIntensityRanged')
    parser.add_argument('--size_x', default=1.5, type=float, 
        help='Pixel size in x direction')
    parser.add_argument('--size_y', default=1.5, type=float, 
        help='Pixel size in y direction')
    parser.add_argument('--size_z', default=2.0, type=float, 
        help='Pixel size in z direction')
    parser.add_argument('--min_iou', default=0, type=float, 
        help='Min. IoU of the 2nd crop with the 1st crop')
    parser.add_argument('--max_iou', default=1.0, type=float, 
        help='Max. IoU of the 2nd crop with the 1st crop')

    # Training params - OK
    parser.add_argument('--use_fp16', action='store_true',
        help='''Whether or not to use half precision for training. 
        Improves training time and memory requirements, but can provoke instability 
        and slight decay of performance. We recommend disabling mixed precision if 
        the loss is unstable, if reducing the patch size or if training with bigger ViTs.''')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
        help='Per-GPU batch size (number of distinct images loaded on one GPU).')
    parser.add_argument('--n_epochs', default=100, type=int, 
        help='Number of epochs of training.')
    parser.add_argument('--base_lr', default=0.0005, type=float, 
        help='''Learning rate at the end of linear warmup (highest used during 
        training). The learning rate is linearly scaled with the batch size, and 
        specified here for a reference batch size of 256.''')
    parser.add_argument('--warmup_epochs', default=10, type=int,
        help='Number of epochs for the linear learning-rate warm up.')
    parser.add_argument('--end_lr', type=float, default=1e-6,
        help='''Target lr at the end of optimization. We use a cosine lr 
        schedule with linear warmup.''')
    parser.add_argument('--base_wd', type=float, default=0.04, 
        help='TODO')
    parser.add_argument('--end_wd', type=float, default=0.4, 
        help='TODO')
    # Increased base_momentum due to smaller batch size
    parser.add_argument('--base_momentum', type=float, default=0.9995,
        help='TODO')

    # Other params
    parser.add_argument('--data_dir', default='./data', type=str,
        help='Path to pretraining data directory.')
    parser.add_argument('--output_dir', default='.', type=str, 
        help='Path to save logs and checkpoints.')
    parser.add_argument('--save_chkpt_every', default=20, type=int, 
        help='How often to save model checkpoint.')
    parser.add_argument('--seed', default=0, type=int, 
        help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, 
        help='Number of data loading workers per GPU.')

    return parser


def train_one_epoch(student, teacher, loss_fn, train_loader,
        optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
        fp16_scaler, device, args):
    losses = []
    tqdm_it = tqdm(train_loader, total=len(train_loader), leave=True)

    for batch_idx, data_dict in enumerate(tqdm_it):
        # Get global step number
        step = len(train_loader) * epoch + batch_idx

        # Set schedulers
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[step]
            if i == 0:  # Only the first group is regularized
                param_group['weight_decay'] = wd_schedule[step]

        # Prepare input
        # TODO: add comment
        x1, x2 = data_dict['img1'].to(device), data_dict['img2'].to(device)
        x_student = x1
        x_teacher = x2
        # x_student = torch.cat([x1, x2])
        # x_teacher = torch.cat([x2, x1])
    
        # Forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            out_student = student(x_student)
            out_teacher = teacher(x_teacher)
            loss = loss_fn(out_student, out_teacher, epoch)

        if not math.isfinite(loss.item()):
            print(f'Loss is {loss.item()}, stopping training...')
            sys.exit(1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update teacher weights
        with torch.no_grad():
            m = momentum_schedule[batch_idx]
            for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
                param_teacher.mul_(m).add_(
                    (1-m) * param_student.detach()
                )

        # Logging
        losses.append(loss.item())
        tqdm_it.set_description(f'Epoch: [{epoch+1}/{args.n_epochs}]')
        tqdm_it.set_postfix(loss=loss.item(), 
                            lr=lr_schedule[step],
                            wd=wd_schedule[step],
                            momentum=momentum_schedule[step])

    return losses


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    dataset = Dataset(data=get_ssl_data(args.data_dir), 
                      transform=get_ssl_transforms(args))
    data_loader = DataLoader(dataset, batch_size=args.batch_size_per_gpu)
    # TODO: num_workers
    # TODO: drop_last?

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
    )

    param_groups = utils.get_param_groups(student)
    optimizer = optim.AdamW(params=param_groups)

    lr_schedule = utils.cosine_scheduler(
        base_val=args.base_lr,
        end_val=args.end_lr,
        n_epochs=args.n_epochs,
        iters_per_epoch=len(data_loader),
        warmup_epochs=args.warmup_epochs
    )
    wd_schedule = utils.cosine_scheduler(
        base_val=args.base_wd,
        end_val=args.end_wd,
        n_epochs=args.n_epochs,
        iters_per_epoch=len(data_loader)
    )
    momentum_schedule = utils.cosine_scheduler(
        base_val=args.base_momentum,
        end_val=1,
        n_epochs=args.n_epochs,
        iters_per_epoch=len(data_loader)
    )

    # Train
    for epoch in range(args.n_epochs):
        train_one_epoch(
            student, teacher, loss_fn, data_loader, optimizer, 
            lr_schedule, wd_schedule, momentum_schedule, epoch, None, device,
            args
        )


if __name__ == '__main__':
    parser = get_args_parser()
    main(parser.parse_args())