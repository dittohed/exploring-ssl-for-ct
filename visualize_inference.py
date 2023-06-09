import argparse

import torch
import matplotlib.pyplot as plt

import src.utils as utils

from functools import partial
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_surface_dice
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

from src.loaders import get_finetune_data
from src.transforms import get_finetune_transforms_2d, get_finetune_transforms_3d


def get_args_parser():
    parser = argparse.ArgumentParser('Finetune CT')

    # Swin params
    parser.add_argument('--embedding_size', default=24, type=int,
        help='Swin backbone base embedding size (C from the paper).')
    
    # Data params
    parser.add_argument('--spatial_dims', default=2, type=int, 
        help='Spatial dimension of input data, either 2 for 2D or 3 for 3D.')
    parser.add_argument('--a_min', default=-500, type=float, 
        help='`a_min` in monai.transforms.ScaleIntensityRanged.')
    parser.add_argument('--a_max', default=500, type=float, 
        help='`a_max` in monai.transforms.ScaleIntensityRanged.')
    parser.add_argument('--size_x', default=1, type=float, 
        help='Pixel size in x direction.')
    parser.add_argument('--size_y', default=1, type=float, 
        help='Pixel size in y direction.')
    parser.add_argument('--size_z', default=2.5, type=float, 
        help='Pixel size in z direction.')
    parser.add_argument('--n_classes', default=14, type=int,
        help='Number of segmentation classes (= number of output channels).')

    # Inference params
    parser.add_argument('--use_amp', action='store_true',
        help='Whether to use Automatic Mixed Precision for inference.')
    parser.add_argument('--sw_batch_size', default=4, type=int,
        help='Batch size for sliding window inference.')
    parser.add_argument('--sw_overlap', default=0.5, type=float,
        help='Sliding window inference overlap.')

    # Other params
    parser.add_argument('--data_dir', default='./data/finetune_preprocessed_2d', type=str,
        help='Path to training data directory.')
    parser.add_argument('--split_path', default='./data/split.json', type=str,
        help='Path to .json file with data split.')
    parser.add_argument('--chkpt_path', type=str, 
        help='Path to model checkpoint.')
    parser.add_argument('--out_dir', default='./out', type=str,
        help='Path to directory for storing inference visualizations.')

    return parser


def save_2d_img_gt_pred_plot(img, label, pred, save_path, args):
    _, axs = plt.subplots(1, 3, figsize=(10, 10))

    axs[0].imshow(img.cpu(), cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Original slice')

    axs[1].imshow(label.cpu(), interpolation='none', 
                  vmin=0, vmax=args.n_classes-1)
    axs[1].set_title('Label')

    axs[2].imshow(pred.cpu(), interpolation='none', 
                  vmin=0, vmax=args.n_classes-1)
    axs[2].set_title('Prediction')

    plt.savefig(save_path)
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, data = get_finetune_data(
        Path(args.data_dir),
        Path(args.split_path)
    )

    # Dummy args
    args.n_crops_per_ct = 0

    if args.spatial_dims == 2:
        _, transforms = get_finetune_transforms_2d(args)
    else:
        _, transforms = get_finetune_transforms_3d(args, device)

    ds = Dataset(
        data=data, 
        transform=transforms
    )
    loader = DataLoader(
        ds, 
        num_workers=0, 
        batch_size=1, 
        shuffle=False
    )

    model = SwinUNETR(
        img_size=tuple([96]*args.spatial_dims),
        in_channels=1,
        out_channels=args.n_classes,
        feature_size=args.embedding_size,
        num_heads=(3, 6, 12, 24),
        spatial_dims=args.spatial_dims
    ).to(device)

    model.load_state_dict(torch.load(args.chkpt_path, map_location=device))
    print(f'Successfully loaded weights from {args.chkpt_path}.')
    model.eval()

    model_infer = partial(
        sliding_window_inference,
        roi_size=tuple([96]*args.spatial_dims),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.sw_overlap
    )

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Prepare stuff for calculating metrics
    acc_fn = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)

    org_thresholds = OrderedDict(  # FLARE2022 official thresholds for surf dice
        {'Liver': 5, 'RK': 3, 'Spleen': 3, 'Pancreas': 5, 
         'Aorta': 2, 'IVC': 2, 'RAG': 2, 'LAG': 2, 'Gallbladder': 2,
         'Esophagus': 3, 'Stomach': 5, 'Duodenum': 7, 'LK': 3}
    )

    avg_dice = utils.AverageAggregator()
    avg_surf_dice = utils.AverageAggregator()

    # Eval loop
    for data_dict in tqdm(loader):
        img, label = data_dict['img'], data_dict['label']

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            with torch.no_grad():
                pred = model_infer(img)

        label_list = decollate_batch(label)
        label_list = [post_label(label_tensor) for label_tensor in label_list]
        pred_list = decollate_batch(pred)
        pred_list = [post_pred(pred_tensor) for pred_tensor in pred_list]

        file_id = Path(data_dict['img_meta_dict']['filename_or_obj'][0]).stem

        # Store visualizations
        if args.spatial_dims == 2:
            img = img[0, 0, :, :]
            label = label[0, 0, :, :]
            pred = torch.argmax(pred[0], dim=0)

            save_path = out_dir / (file_id + '.png')
            save_2d_img_gt_pred_plot(img, label, pred, save_path, args)
        else:
            for i in range(img.shape[-1]):  
                img = img[0, 0, :, :, i]
                label = label[0, 0, :, :, i]
                pred = torch.argmax(pred[0, :, :, :, i], dim=0)

                save_path = out_dir / (file_id + f'_{i}.png')
                save_2d_img_gt_pred_plot(img, label, pred, save_path, args)

        # Dice
        acc_fn.reset()
        acc_fn(y_pred=pred_list, y=label_list)
        acc, not_nans = acc_fn.aggregate()
        assert not_nans == 1
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

    print(f'Mean validation dice score: {avg_dice.item():.4f}')
    print(f'Mean validation surface dice score: {avg_surf_dice.item():.4f}.')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
