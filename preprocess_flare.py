import argparse

from pathlib import Path
from monai.data import DataLoader, Dataset, NibabelWriter

from src.loaders import get_ssl_data
from src.transforms import get_ssl_transforms


def get_args_parser():
    parser = argparse.ArgumentParser('Preprocess CT')

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
    
    parser.add_argument('--data_dir', default='./data/ssl', type=str,
        help='Path to pretraining data directory.')
    parser.add_argument('--output_dir', default='./data/ssl_preprocessed', type=str, 
        help='Path to save logs and checkpoints.')

    return parser


def main(args):
    data = get_ssl_data(args.data_dir)
    transforms = get_ssl_transforms(args, mode='preprocess')

    ds = Dataset(data=data, transform=transforms)
    loader = DataLoader(ds, batch_size=1)

    writer = NibabelWriter()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for data in loader:
        load_path = Path(data['img_meta_dict']['filename_or_obj'][0])
        load_name = load_path.name.split('.')[0]
        save_path = args.output_dir / Path(f'{load_name}_processed.nii.gz')

        writer.set_data_array(data['img'][0], channel_dim=0)
        writer.write(save_path, verbose=False)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)