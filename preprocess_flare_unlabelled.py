import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from monai.data import DataLoader, Dataset, NibabelWriter

from src.loaders import get_ssl_data
from src.transforms import get_preprocess_transforms_2d, get_preprocess_transforms_3d


def get_args_parser():
    parser = argparse.ArgumentParser('Preprocess unlabelled CTs from FLARE')

    parser.add_argument('--spatial_dims', default=2, type=int, 
        help='''Spatial dimension of output data, either 2 for 2D (separate .png
        slices will be saved) or 3 for 3D (.nii.gz volumes will be saved).''')
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
        help='Path to original pretraining data directory.')
    parser.add_argument('--output_dir', default='./data/ssl_preprocessed_2d', type=str, 
        help='Path to save preprocessed data.')

    return parser


def main(args):
    data = get_ssl_data(args.data_dir)

    if args.spatial_dims == 2:
        transforms = get_preprocess_transforms_2d(args, mode='ssl')
    else:
        transforms = get_preprocess_transforms_3d(args)
        writer = NibabelWriter()

    ds = Dataset(data=data, transform=transforms)
    loader = DataLoader(ds, batch_size=1)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for data in tqdm(loader):
        load_path = Path(data['img_meta_dict']['filename_or_obj'][0])
        load_name = load_path.name.split('.')[0]

        if args.spatial_dims == 2:
            n_slices = data['img'][0].shape[-1]
            for i in range(n_slices):
                img_slice = data['img'][0][0, :, :, i].numpy()
                img_slice = Image.fromarray((img_slice * 255).astype('uint8'))

                save_path = args.output_dir / Path(f'{load_name}_{i}.png')
                img_slice.save(save_path)
                
        elif args.spatial_dims == 3:
            save_path = args.output_dir / Path(f'{load_name}.nii.gz')

            writer.set_data_array(data['img'][0], channel_dim=0)
            writer.write(save_path, verbose=False)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)