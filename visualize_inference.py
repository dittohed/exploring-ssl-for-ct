import argparse
import sys 

import torch
import matplotlib.pyplot as plt

import src.utils as utils

from functools import partial
from pathlib import Path

from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

from src.loaders import get_finetune_data
from src.transforms import get_finetune_transforms


def get_args_parser():
    parser = argparse.ArgumentParser('Finetune CT')

    # Swin params
    parser.add_argument('--embedding_size', default=48, type=int,
        help='Swin backbone base embedding size (C from the paper).')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',  # TODO: could try
        help='Whether to use gradient checkpointing (saves memory, longer training).')
    
    # Data params
    parser.add_argument('--spatial_dims', default=3, type=int, 
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
        help='Whether to use AMP for training.')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
        help='`num_samples` in monai.transforms.RandCropByPosNegLabeld (per GPU).')
    parser.add_argument('--sw_batch_size', default=4, type=int,
        help='Batch size for sliding window inference.')
    parser.add_argument('--sw_overlap', default=0.5, type=float,
        help='Sliding window inference overlap.')

    # Other params
    parser.add_argument('--data_dir', default='./data/finetune', type=str,
        help='Path to training data directory.')
    parser.add_argument('--chkpt_path', type=str, 
        help='Path to model checkpoint.')

    return parser


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, data = get_finetune_data(args.data_dir)
    _, transforms = get_finetune_transforms(args, device)

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
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=args.n_classes,
        feature_size=args.embedding_size,
        num_heads=(3, 6, 12, 24)
    ).to(device)

    model.load_state_dict(torch.load(args.chkpt_path, map_location=device))
    model.eval()

    model_infer = partial(
        sliding_window_inference,
        roi_size=[96, 96, 96],
        sw_batch_size=1,
        predictor=model,
        overlap=0.5
    )

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    acc_fn = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)

    avg_agg = utils.AverageAggregator()

    for data_dict in loader:
        img, label = data_dict['img'], data_dict['label']

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            with torch.no_grad():
                pred = model_infer(img)

        label_list = decollate_batch(label)
        label_list = [post_label(label_tensor) for label_tensor in label_list]
        pred_list = decollate_batch(pred)
        pred_list = [post_pred(pred_tensor) for pred_tensor in pred_list]

        # Store visualizations
        for i in range(img.shape[-1]):
            if i % 50 == 0:
                fig, axs = plt.subplots(1, 3, figsize=(10, 10))
                axs[0].imshow(img[0, 0, :, :, i].cpu(), cmap='gray', vmin=0, vmax=1)
                axs[0].set_title('Original slice')
                axs[1].imshow(label[0, 0, :, :, i].cpu())
                axs[1].set_title('Label')
                axs[2].imshow(torch.argmax(pred[0, :, :, :, i].cpu(), dim=0))
                axs[2].set_title('Prediction')

                file_id = Path(data_dict['img_meta_dict']['filename_or_obj'][0]).stem
                plt.savefig(f'{file_id}_{i}.png')
                plt.close()

        acc_fn.reset()
        acc_fn(y_pred=pred_list, y=label_list)
        acc, not_nans = acc_fn.aggregate()
        assert not_nans == 1
        avg_agg.update(acc.item())

    print(f'Mean validation dice score: {avg_agg.item():.4f}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    main(args)
