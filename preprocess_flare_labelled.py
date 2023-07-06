import argparse

from pathlib import Path
from PIL import Image
from monai.data import DataLoader, Dataset

from src.loaders import get_finetune_data
from src.transforms import get_preprocess_transforms_2d


def get_args_parser():
    parser = argparse.ArgumentParser('Preprocess labelled CTs from FLARE to 2D')

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
    
    parser.add_argument('--data_dir', default='./data/finetune', type=str,
        help='Path to original labelled data directory.')
    parser.add_argument('--output_dir', default='./data/finetune_preprocessed_2d', type=str, 
        help='Path to save preprocessed data.')

    return parser


def main(args):
    train_data, val_data = get_finetune_data(args.data_dir)
    transforms = get_preprocess_transforms_2d(args, mode='finetune')

    ds = Dataset(data=train_data+val_data, transform=transforms)
    loader = DataLoader(ds, batch_size=1)

    output_dir_imgs = Path(args.output_dir)/Path('imgs')
    output_dir_labels = Path(args.output_dir)/Path('labels')
    output_dir_imgs.mkdir(parents=True, exist_ok=True)
    output_dir_labels.mkdir(parents=True, exist_ok=True)

    for data in loader:
        n_slices = data['img'][0].shape[-1]

        for i in range(n_slices):
            img_slice = data['img'][0][0, :, :, i].numpy()
            img_slice = Image.fromarray((img_slice * 255).astype('uint8'))

            img_path = Path(data['img_meta_dict']['filename_or_obj'][0])
            img_name = img_path.name.split('.')[0]
            img_slice.save(output_dir_imgs/Path(f'{img_name}_{i}.png'))

            label_slice = data['label'][0][0, :, :, i].numpy()
            label_slice = Image.fromarray(label_slice.astype('uint8'))

            label_path = Path(data['label_meta_dict']['filename_or_obj'][0])
            label_name = label_path.name.split('.')[0]
            label_slice.save(output_dir_labels/Path(f'{label_name}_{i}.png'))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)