from pathlib import Path


def get_ssl_data(data_dir: str):
    img_paths = Path(data_dir).glob('*.nii.gz')
    data = [{'img': img_path} for img_path in img_paths]
    return data