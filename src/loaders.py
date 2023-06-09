import json
from pathlib import Path


def get_ssl_data(data_dir: str):
    img_paths = Path(data_dir).glob('*.nii.gz')
    data = [{'img': img_path} for img_path in img_paths]
    return data


def get_finetune_data(data_dir: str):
    data_dir = Path(data_dir)
    with open(data_dir/'split.json') as json_file:
        split = json.load(json_file)

    train_data = [
        {
            'img': data_dir/Path(f'imgs/{ct_id}_0000.nii.gz'),
            'label': data_dir/Path(f'labels/{ct_id}.nii.gz')    
        } 
        for ct_id in split['train']
    ]

    val_data = [
        {
            'img': data_dir/Path(f'imgs/{ct_id}_0000.nii.gz'),
            'label': data_dir/Path(f'labels/{ct_id}.nii.gz')    
        } 
        for ct_id in split['val']
    ]

    return train_data, val_data