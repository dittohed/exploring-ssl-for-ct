import numpy as np

from typing import Optional

from monai.config import KeysCollection
from monai.utils import first
from monai.transforms import (
    Crop,
    Randomizable,
    MapTransform,
    LoadImaged,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    Orientationd,
    Spacingd,
    ToTensord
)


# TODO: doublecheck, verify there's no coords issue
# TODO: add type hints and args
class IoUCropd(Randomizable, MapTransform):
    """
    MONAI-style transformation implementing IoU-based cropping.

    Firstly, a random subvolume is cropped, then another one so that
    its IoU with the first one is within [`min_iou`, `max_iou`] range.

    Both subvolumes are cubes with side lengths of `crop_size`.

    Args:
        keys (monai.config.KeysCollection): keys of a data dictionary for which
            the transformation should be applied.
        crop_size (int): subvolumes side lenghts.
        min_iou (float): lower bound of IoU range.
        max_iou (float): upper bound of IoU range.
        debug (bool): whether to add/leave keys to the data dictionary
            for debugging purposes.
    """

    def __init__(
            self, keys: KeysCollection, crop_size: int = 96, 
            min_iou: float = 0.0, max_iou: float = 1.0, 
            debug=False):
        assert min_iou <= max_iou

        super(IoUCropd, self).__init__(keys)

        self._crop_size = crop_size
        self._min_iou = min_iou
        self._max_iou = max_iou
        self._debug = debug
        self._cropper = Crop()

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> 'IoUCropd':
        super().set_random_state(seed, state)
        return self

    def randomize(self, shape) -> tuple:
        # Leave spatial dims only and enforce x, y, z order 
        # instead of y, x, z order
        shape = list(shape[-3:])
        shape[0], shape[1] = shape[1], shape[0]

        # Sample first crop
        existing_coords = self._sample_coords(
                            crop_size=self._crop_size, 
                            limit_box=[0, 0, 0, *shape]
                        )

        # Sample second crop
        iou_coords = self._sample_coords_iou(
                    existing_coords=existing_coords,
                    crop_size=self._crop_size,
                    img_shape=shape)
        
        return existing_coords, iou_coords

    def _sample_coords(self, crop_size, limit_box):
        """
        Sample minimum coord (x1, y1, z1) and maximum coord (x2, y2, z2) 
        of a cube crop with a side length of `crop_size` so that it fully  
        stays within `limit_box` (prevents cropping outside of a desired volume).
        """

        x1 = self.R.randint(limit_box[0], limit_box[3]-crop_size+1)
        y1 = self.R.randint(limit_box[1], limit_box[4]-crop_size+1)
        z1 = self.R.randint(limit_box[2], limit_box[5]-crop_size+1)

        return x1, y1, z1, x1+crop_size, y1+crop_size, z1+crop_size
    
    def _sample_coords_iou(self, existing_coords, crop_size, img_shape):
        """
        Sample minimum coord (x1, y1, z1) and maximum coord (x2, y2, z2) 
        of a cube crop with a side length of `crop_size` so that it has
        IoU from a specified range with a cube crop defined by 
        `existing_coords`.
        """

        while True:
            if self._min_iou > 0:
                # Speed up by restricting only to volume 
                # adjacent to `existing_coords` crop
                limit_box = (
                    max(existing_coords[0]-crop_size, 0),
                    max(existing_coords[1]-crop_size, 0),
                    max(existing_coords[2]-crop_size, 0),
                    min(existing_coords[3]+crop_size, img_shape[0]),
                    min(existing_coords[4]+crop_size, img_shape[1]),
                    min(existing_coords[5]+crop_size, img_shape[2])
                )
            else:
                limit_box = [0, 0, 0, *img_shape]

            new_coords = self._sample_coords(crop_size, limit_box)
            iou = self.iou3d(existing_coords, new_coords)
            if iou >= self._min_iou and iou <= self._max_iou:
                break 

        return new_coords

    def __call__(self, data):
        coords1, coords2 = self.randomize(data[first(self.keys)].shape)

        d = dict(data)
        for key in self.keys:
            d[f'{key}1'] = self._cropper(d[key], self.coords_to_slices(coords1))
            d[f'{key}2'] =  self._cropper(d[key], self.coords_to_slices(coords2))

            if self._debug:
                d[f'{key}1_coords'] = coords1 
                d[f'{key}2_coords'] = coords2
            else: 
                del d[key]

        return d
    
    @staticmethod
    def coords_to_slices(coords):
        """
        Convert [x1, y1, z1, x2, y2, z2] list denoting minimum and maximum coords 
        of a cube crop to [y1 : y2, x1 : x2, z1 : z2] slice for monai.transforms.Crop.
        """ 

        slices = [
            slice(coords[1], coords[4]),
            slice(coords[0], coords[3]),
            slice(coords[2], coords[5])
        ]

        return slices

    @staticmethod
    def iou3d(a, b):
        """
        Calcute IoU between cube crops `a` and `b`, where each is 
        a list [x1, y1, z1, x2, y2, z2] denoting minimum and maximum coords.
        """

        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        z1 = max(a[2], b[2])
        x2 = min(a[3], b[3])
        y2 = min(a[4], b[4])
        z2 = min(a[5], b[5])

        width = x2 - x1
        depth = y2 - y1
        height = z2 - z1

        # Not intersecting
        if width <= 0 or depth <= 0 or height <= 0:
            return 0
        
        vol_inter = width * depth * height
        vol_a = (a[3]-a[0]) * (a[4]-a[1]) * (a[5]-a[2])
        vol_b = (b[3]-b[0]) * (b[4]-b[1]) * (b[5]-b[2])
        vol_union = vol_a + vol_b - vol_inter

        return vol_inter / vol_union
    

def get_ssl_transforms(args):
    transforms = Compose([
        LoadImaged(keys=['img']),
        EnsureChannelFirstd(keys=['img']),
        Orientationd(keys=['img'], axcodes='RAS'),
        # TODO: double-check pixdim order below
        Spacingd(keys=['img'], pixdim=(args.size_y, args.size_x, args.size_z), 
                 mode=('bilinear')),
        ScaleIntensityRanged(keys=['img'], a_min=args.a_min, a_max=args.a_max,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['img'], source_key='img'),
        EnsureTyped(keys=['img', 'label'], track_meta=False),
        IoUCropd(keys=['img'], min_iou=args.min_iou, max_iou=args.max_iou)
        # TODO: put augmentations here
    ])

    return transforms


def get_finetune_transforms(args):
    base_transforms = [
        LoadImaged(keys=['img', 'label']),
        EnsureChannelFirstd(keys=['img', 'label']),
        Orientationd(keys=['img', 'label'], axcodes='RAS'),
        # TODO: double-check pixdim order below
        Spacingd(keys=['img', 'label'], pixdim=(args.size_y, args.size_x, args.size_z), 
                    mode=('bilinear', 'nearest')),
        ScaleIntensityRanged(keys=['img'], a_min=args.a_min, a_max=args.a_max,
                                b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['img', 'label'], source_key='img'),
        EnsureTyped(keys=['img', 'label'], track_meta=False)
    ]

    train_transforms = Compose([
        *base_transforms,
        RandCropByPosNegLabeld(  # TODO: doublecheck this transform
            keys=['img', 'label'],
            label_key='label',
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=args.batch_size_per_gpu,
            image_key='img',
            image_threshold=0
        )
        # # TODO: put augmentations here
    ])

    val_transforms = Compose([*base_transforms])

    return train_transforms, val_transforms