import json
from pathlib import Path


def get_ssl_data(data_dir: str):
    img_paths = sorted(list(Path(data_dir).glob('*')))
    data = [{'img': img_path} for img_path in img_paths]
    return data


def get_finetune_data(data_dir: Path, split_path: Path):
    with open(split_path) as json_file:
        split = json.load(json_file)

    imgs = sorted(list(Path(data_dir/'imgs').glob('*')))
    labels = sorted(list(Path(data_dir/'labels').glob('*')))

    assert len(imgs) == len(labels)

    train_data = []
    val_data = []

    for img, label in zip(imgs, labels):
        ct_id_img = '_'.join(img.name.split('.')[0].split('_')[:3])
        ct_id_label = '_'.join(label.name.split('.')[0].split('_')[:3])

        assert ct_id_img == ct_id_label

        if ct_id_img in split['train']:
            train_data.append(
                {'img': img, 'label': label}
            )
        elif ct_id_img in split['val']:
            val_data.append(
                {'img': img, 'label': label}
            )

    return train_data, val_data