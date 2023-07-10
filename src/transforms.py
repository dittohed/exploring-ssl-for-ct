import numpy as np
import monai.transforms as T

from typing import Optional

from monai.config import KeysCollection
from monai.utils import first


# TODO: add type hints
class IoUCropd(T.Randomizable, T.MapTransform):
    """
    MONAI-style transformation implementing IoU-based cropping.

    Firstly, a random patch is cropped, then another one so that
    its IoU with the first one is within [`min_iou`, `max_iou`] range.

    Both resulting crops are squares/cubes with side lengths of `crop_size`.

    Args:
        keys (monai.config.KeysCollection): keys of a data dictionary for which
            the transformation should be applied.
        spatial_dims (int): input spatial dimensions, either 2 for 2D or 3 for 3D.
        crop_size (int): side lenghts.
        min_iou (float): lower bound of IoU range.
        max_iou (float): upper bound of IoU range.
        max_retries_existing (int): how many times to sample second crop coords
            having sampled first crop coords. Introduced to not sample first
            crop coords every time IoU isn't in the specified range to
            decrease chances of avoiding harder first central crops.
        max_retries_total (int): how many times to sample in general. If
            proper pair isn't find in `max_retries_total` steps, random crops
            are returned.
        debug (bool): whether to add/leave keys to the data dictionary
            for debugging purposes.
    """

    def __init__(
            self, keys: KeysCollection, spatial_dims: int, crop_size: int = 96, 
            min_iou: float = 0.0, max_iou: float = 1.0, 
            max_retries_existing: int = 20,
            max_retries_total: int = 60, debug=False):
        assert spatial_dims in [2, 3]
        assert min_iou <= max_iou

        super(IoUCropd, self).__init__(keys)

        self._crop_size = crop_size
        self._spatial_dims = spatial_dims
        self._min_iou = min_iou
        self._max_iou = max_iou
        self._max_retries_existing = max_retries_existing
        self._max_retries_total = max_retries_total
        self._debug = debug

        self._cropper = T.Crop()

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> 'IoUCropd':
        super().set_random_state(seed, state)
        return self
    
    def _get_limit_box(self, shape: list) -> list:
        """
        Return a list of format [x1, y1, ..., x2, y2, ...]
        denoting a rectangle/volume covering whole original image of shape 
        `shape` in format [x1, y1, ...].
        """

        limit_box = [
            *list(np.zeros(self._spatial_dims, dtype=int)),
            *shape
        ]

        return limit_box

    def randomize(self, shape) -> tuple:
        # Leave spatial dims only and enforce x-first order
        # instead of y-first order
        # TODO: this is probably not needed, simplify
        shape = list(shape[-self._spatial_dims:])
        shape[0], shape[1] = shape[1], shape[0]

        retries_total = 0
        while True:
            # Sample first crop
            existing_coords = self._sample_coords(
                crop_size=self._crop_size, 
                limit_box=self._get_limit_box(shape)
            )
            
            # Sample second crop
            iou_coords, success = self._sample_coords_iou(
                existing_coords=existing_coords,
                crop_size=self._crop_size,
                img_shape=shape,
                max_retries=self._max_retries_existing
            )
            if success:
                return existing_coords, iou_coords
            else:
                retries_total += self._max_retries_existing
                
            if retries_total >= self._max_retries_total:
                print(
                    f'''Pair not found within {self._max_retries_total} steps, '''
                    '''returning random crops.'''
                )
                break

        return existing_coords, iou_coords

    def _sample_coords(self, crop_size: int, limit_box: list) -> tuple:
        """
        Sample minimum coord in format [x1, y1, ...] and maximum coord 
        in format [x2, y2, ...] of a crop with a side length `crop_size` so that 
        it fully stays within `limit_box` (prevents cropping outside of a 
        desired rectangle/volume).
        """

        if self._spatial_dims == 2:
            x1 = self.R.randint(limit_box[0], limit_box[2]-crop_size+1)
            y1 = self.R.randint(limit_box[1], limit_box[3]-crop_size+1)
            coords = (x1, y1, x1+crop_size, y1+crop_size)
        else:
            x1 = self.R.randint(limit_box[0], limit_box[3]-crop_size+1)
            y1 = self.R.randint(limit_box[1], limit_box[4]-crop_size+1)
            z1 = self.R.randint(limit_box[2], limit_box[5]-crop_size+1)
            coords = (x1, y1, z1, x1+crop_size, y1+crop_size, z1+crop_size)
        
        return coords
    
    def _sample_coords_iou(self, existing_coords, crop_size, img_shape,
            max_retries):
        """
        Sample minimum coord in format [x1, y1, ...] and maximum coord
        in format [x2, y2, ...] of a crop with a side length of `crop_size` so 
        that it has IoU from a specified range with a crop defined by 
        `existing_coords`.

        Returns `new_coords` with True flag if the new coords were found
        within `max_retries` steps, otherwise False flag is returned.
        """
        
        if self._min_iou > 0:
            # Speed up by restricting only to square/volume 
            # adjacent to `existing_coords` crop
            if self._spatial_dims == 2:
                limit_box = (
                    max(existing_coords[0]-crop_size, 0),
                    max(existing_coords[1]-crop_size, 0),
                    min(existing_coords[2]+crop_size, img_shape[0]),
                    min(existing_coords[3]+crop_size, img_shape[1])
                ) 
            else:
                limit_box = (
                    max(existing_coords[0]-crop_size, 0),
                    max(existing_coords[1]-crop_size, 0),
                    max(existing_coords[2]-crop_size, 0),
                    min(existing_coords[3]+crop_size, img_shape[0]),
                    min(existing_coords[4]+crop_size, img_shape[1]),
                    min(existing_coords[5]+crop_size, img_shape[2])
                )
        else:
            limit_box = self._get_limit_box(img_shape)

        for _ in range(max_retries):
            new_coords = self._sample_coords(crop_size, limit_box)
            iou = self._iou(existing_coords, new_coords)
            if iou >= self._min_iou and iou <= self._max_iou:
                return new_coords, True

        return new_coords, False

    def __call__(self, data):
        coords1, coords2 = self.randomize(data[first(self.keys)].shape)

        d = dict(data)
        for key in self.keys:
            d[f'{key}1'] = self._cropper(d[key], self._coords_to_slices(coords1))
            d[f'{key}2'] =  self._cropper(d[key], self._coords_to_slices(coords2))

            if self._debug:
                d[f'{key}1_slices'] = str(self._coords_to_slices(coords1))
                d[f'{key}2_slices'] = str(self._coords_to_slices(coords2))
            else: 
                del d[key]  # Remove original image

        return d
    
    def _coords_to_slices(self, coords):
        """
        Convert [x1, y1, ..., x2, y2, ...] list denoting first minimum, then 
        maximum coords of a crop to [y1 : y2, x1 : x2, ...] slice 
        for monai.transforms.Crop.
        """ 

        if self._spatial_dims == 2:
            slices = [
                slice(coords[1], coords[3]),
                slice(coords[0], coords[2]),
            ]
        else:
            slices = [
                slice(coords[1], coords[4]),
                slice(coords[0], coords[3]),
                slice(coords[2], coords[5])
            ]

        return slices

    def _iou(self, a, b):
        """
        Calcute IoU between crops `a` and `b`, where each is 
        a list [x1, y1, ..., x2, y2, ...] denoting minimum and maximum coords.
        """

        if self._spatial_dims == 2:
            iou = self.iou2d(a, b)
        else:
            iou = self.iou3d(a, b)

        return iou

    @staticmethod
    def iou2d(a, b):
        """
        Calcute IoU between rectangular crops `a` and `b`, where each is 
        a list [x1, y1, x2, y2] denoting minimum and maximum coords.
        """

        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        width = x2 - x1
        height = y2 - y1

        # Not intersecting
        if width <= 0 or height <= 0:
            return 0
        
        area_inter = width * height
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        area_union = area_a + area_b - area_inter

        return area_inter / area_union

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
    

def get_ssl_rand_transforms(key: str, spatial_dims: int) -> list:
    """
    Return random img transforms to be applied on a single image
    from a positive pair.
    """

    transforms = [
        T.RandZoomd(
            keys=[key], 
            min_zoom=0.8, 
            max_zoom=1.2, 
            mode=('bilinear'),
            padding_mode='constant',
            prob=0.75
        ),
        T.RandFlipd(
            keys=[key], 
            spatial_axis=0,
            prob=0.5
        ),
        T.RandFlipd(
            keys=[key], 
            spatial_axis=1,
            prob=0.5
        )
    ]

    if spatial_dims == 3:
        transforms.append(
            T.RandFlipd(
                keys=[key], 
                spatial_axis=2,
                prob=0.5
            )
        )

    transforms.extend([
        T.RandRotate90d(
            keys=[key], 
            max_k=3,
            prob=0.5
        ),
        T.RandScaleIntensityd(
            keys=[key], 
            factors=0.1, 
            prob=0.75
        ),
        T.RandShiftIntensityd(
            keys=[key], 
            offsets=0.1, 
            prob=0.75
        ),
        T.RandGaussianSmoothd(
            keys=[key],
            prob=0.25
        ),
        T.RandGaussianNoised(
            keys=[key],
            std=0.01,
            prob=0.25
        )
    ])

    return transforms


def get_preprocess_transforms_2d(args, mode='finetune') -> T.Compose:
    """
    Return img transforms for 2D preprocessing within 
    a preprocessing script (but takes 3D input).

    Resulting images should be further loaded using 
    `get_finetune_transforms_2d` if using 'finetune' mode or 
    `get_ssl_transforms_2d` if using 'ssl' mode.
    """
    
    assert mode in ['finetune', 'ssl']
    keys = ['img'] if mode == 'ssl' else ['img', 'label']
    interpol = ('bilinear') if mode == 'ssl' else ('bilinear', 'nearest')

    transforms = [
        T.LoadImaged(
            keys=keys
        ),
        T.EnsureChannelFirstd(
            keys=keys
        ),
        T.Orientationd(
            keys=keys, 
            axcodes='RAS'
        ),
        T.Spacingd(
            keys=keys, 
            pixdim=(args.size_y, args.size_x, args.size_z),
            mode=interpol
        ),
        T.ScaleIntensityRanged(
            keys=['img'], 
            a_min=args.a_min, 
            a_max=args.a_max,
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        T.CropForegroundd(
            keys=keys, 
            source_key='img'
        ),
        T.SpatialPadd(
            keys=keys, 
            spatial_size=(96, 96, -1)
        )
    ]

    print(f'The following transforms pipeline will be used: {transforms}.')
    return T.Compose(transforms)


def get_ssl_transforms_2d(args, debug=False) -> T.Compose:
    """
    Return img transforms for 2D pretraining.

    Input images should be first processed using 
    `get_preprocess_transforms_2d` within a preprocessing script.
    """

    transforms = [
        T.LoadImaged(
            keys=['img'],
            reverse_indexing=False  # Don't apply any auto rotation when loading
        ),
        T.ScaleIntensityRanged(  # 2D images stored in uint8
            keys=['img'],
            a_min=0,
            a_max=255,
            b_min=0.0, 
            b_max=1.0,
        ),
        T.EnsureChannelFirstd(
            keys=['img']
        ),
        T.EnsureTyped(
            keys=['img'], 
            track_meta=False
        ),
        IoUCropd(
            keys=['img'], 
            spatial_dims=2,
            min_iou=args.min_iou, 
            max_iou=args.max_iou,
            debug=debug
        ),
        T.EnsureTyped(
            keys=['img1', 'img2'], 
            track_meta=False
        ),
        *get_ssl_rand_transforms('img1', spatial_dims=2),
        *get_ssl_rand_transforms('img2', spatial_dims=2)
    ]

    print(f'The following transforms pipeline will be used: {transforms}.')
    return T.Compose(transforms)


def get_finetune_transforms_2d(args) -> tuple:
    """
    Return img transforms for 2D fine-tuning.

    Input images should be first processed using 
    `get_preprocess_transforms_2d` within a preprocessing script.
    """

    # Same for trainining and validation
    base_transforms = [
        T.LoadImaged(
            keys=['img', 'label'],
            reverse_indexing=False  # Don't apply any auto rotation when loading
        ),
        T.ScaleIntensityRanged(  # 2D images stored in uint8
            keys=['img'],
            a_min=0,
            a_max=255,
            b_min=0.0, 
            b_max=1.0,
        ),
        T.EnsureChannelFirstd(
            keys=['img', 'label']
        ),
        T.EnsureTyped(
            keys=['img', 'label'], 
            track_meta=False
        )
    ]

    train_transforms = [
        *base_transforms,
        T.RandCropByPosNegLabeld(  # Will only raise warning for only-bg labels
            keys=['img', 'label'],
            label_key='label',
            spatial_size=(96, 96),
            pos=1,
            neg=1,
            num_samples=args.n_crops_per_ct
        ),
        T.RandGaussianSmoothd(
            keys=['img'],
            prob=0.15
        ),
        T.RandScaleIntensityd(
            keys=['img'], 
            factors=0.1, 
            prob=0.15
        ),
        T.RandShiftIntensityd(
            keys=['img'], 
            offsets=0.1, 
            prob=0.15
        ),
        T.RandGaussianNoised(
            keys=['img'],
            std=0.01,
            prob=0.15
        ),
        T.RandFlipd(
            keys=['img', 'label'], 
            spatial_axis=0,
            prob=0.25
        ),
        T.RandFlipd(
            keys=['img', 'label'], 
            spatial_axis=1,
            prob=0.25
        ),
        T.RandRotate90d(
            keys=['img', 'label'], 
            max_k=3,
            prob=0.25
        )
    ]

    print(f'''The following transforms pipeline will be used 
              for training: {train_transforms}.''')
    print(f'''The following transforms pipeline will be used 
              for validation: {base_transforms}.''')
    
    return (T.Compose(train_transforms), T.Compose(base_transforms))


def get_preprocess_transforms_3d(args) -> T.Compose:
    """
    Return img transforms for 3D preprocessing within 
    a preprocessing script.

    Resulting images should be further loaded using 
    `get_ssl_transforms_3d`. Preprocessing for 3D finetuning isn't
    implemented as for 2D, using `monai.data.PersistentDataset` instead should
    be enough.
    """

    transforms = [
        T.LoadImaged(
            keys=['img']
        ),
        T.EnsureChannelFirstd(
            keys=['img']
        ),
        T.Orientationd(
            keys=['img'], 
            axcodes='RAS'
        ),
        T.Spacingd(
            keys=['img'], 
            pixdim=(args.size_y, args.size_x, args.size_z), 
            mode=('bilinear')
        ),
        T.ScaleIntensityRanged(
            keys=['img'], 
            a_min=args.a_min, 
            a_max=args.a_max,
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        T.CropForegroundd(
            keys=['img'], 
            source_key='img'
        ),
        T.SpatialPadd(
            keys=['img'], 
            spatial_size=(96, 96, 96)
        ),
        T.EnsureTyped(
            keys=['img'], 
            track_meta=False
        )
    ]

    print(f'The following transforms pipeline will be used: {transforms}.')
    return T.Compose(transforms)


def get_ssl_transforms_3d(args, device=None, debug=False):
    """
    Return img transforms for 3D pretraining.

    Input images should be first processed using 
    `get_preprocess_transforms_3d` within a preprocessing script.
    """

    transforms = [
        T.LoadImaged(
            keys=['img']
        ),
        T.EnsureChannelFirstd(
            keys=['img']
        ),
        T.EnsureTyped(  
            keys=['img'], 
            track_meta=False
        ),
        IoUCropd(
            keys=['img'], 
            spatial_dims=3,
            min_iou=args.min_iou, 
            max_iou=args.max_iou,
            debug=debug
        ),
        T.EnsureTyped(
            keys=['img1', 'img2'], 
            track_meta=False,
            device=device
        ),
        *get_ssl_rand_transforms('img1', spatial_dims=3),
        *get_ssl_rand_transforms('img2', spatial_dims=3)
    ]

    print(f'The following transforms pipeline will be used: {transforms}.')
    return T.Compose(transforms)


def get_finetune_transforms_3d(args, device=None):
    """
    Return img transforms for 3D fine-tuning.

    Should be used for training with 3D original, unprocessed data.
    """

    # Same for trainining and validation
    base_transforms = [
        T.LoadImaged(
            keys=['img', 'label']
        ),
        T.EnsureChannelFirstd(
            keys=['img', 'label']
        ),
        T.Orientationd(
            keys=['img', 'label'], 
            axcodes='RAS'
        ),
        T.Spacingd(
            keys=['img', 'label'], 
            pixdim=(args.size_y, args.size_x, args.size_z),
            mode=('bilinear', 'nearest')
        ),
        T.ScaleIntensityRanged(
            keys=['img'], 
            a_min=args.a_min, 
            a_max=args.a_max,
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        T.CropForegroundd(
            keys=['img', 'label'], 
            source_key='img'
        ),
        T.SpatialPadd(
            keys=['img', 'label'], 
            spatial_size=(96, 96, 96)
        )
    ]

    train_transforms = [
        *base_transforms,
        T.FgBgToIndicesd(
            keys='label',
            fg_postfix='_fg',
            bg_postfix='_bg',
            image_key='img',
        ),
        T.EnsureTyped(
            keys=['img', 'label'], 
            track_meta=False,
            device=device  # Allows performing remaining transforms on GPU
        ),
        T.RandCropByPosNegLabeld(
            keys=['img', 'label'],
            label_key='label',
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=args.n_crops_per_ct,
            fg_indices_key='label_fg',
            bg_indices_key='label_bg'
        ),
        T.RandGaussianSmoothd(
            keys=['img'],
            prob=0.15
        ),
        T.RandScaleIntensityd(
            keys=['img'], 
            factors=0.1, 
            prob=0.15
        ),
        T.RandShiftIntensityd(
            keys=['img'], 
            offsets=0.1, 
            prob=0.15
        ),
        T.RandGaussianNoised(
            keys=['img'],
            std=0.01,
            prob=0.15
        ),
        T.RandFlipd(
            keys=['img', 'label'], 
            spatial_axis=0,
            prob=0.25
        ),
        T.RandFlipd(
            keys=['img', 'label'], 
            spatial_axis=1,
            prob=0.25
        ),
        T.RandFlipd(
            keys=['img', 'label'], 
            spatial_axis=2,
            prob=0.25
        ),
        T.RandRotate90d(
            keys=['img', 'label'], 
            max_k=3,
            prob=0.25
        )
    ]

    val_transforms = [
        *base_transforms,
        T.EnsureTyped(
            keys=['img', 'label'], 
            track_meta=False,
            device=device
        )
    ]

    print(f'''The following transforms pipeline will be used 
              for training: {train_transforms}.''')
    print(f'''The following transforms pipeline will be used 
              for validation: {val_transforms}.''')
    
    return (T.Compose(train_transforms), T.Compose(val_transforms))