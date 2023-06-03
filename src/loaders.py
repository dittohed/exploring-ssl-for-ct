from pathlib import Path


def get_ssl_data(data_dir: str):
    img_paths = Path(data_dir).glob('*.nii.gz')
    data = [{'img': img_path} for img_path in img_paths]
    return data


def get_finetune_data(data_dir: str):
    img_paths = sorted((Path(data_dir)/Path('imgs')).glob('*.nii.gz'))
    label_paths = sorted((Path(data_dir)/Path('labels')).glob('*.nii.gz'))

    assert len(img_paths) == len(label_paths)

    data = [{'img': img_path, 'label': label_path} 
        for img_path, label_path in zip(img_paths, label_paths)]

    return data[:40], data[40:]  # TODO: use true split