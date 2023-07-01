import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


class DataAugmentation:
    """Create crops of an input image together with additional augmentation.

    It generates 2 global crops and `n_local_crops` local crops.

    Parameters
    ----------
    global_crops_scale : tuple
        Range of sizes for the global crops.

    local_crops_scale : tuple
        Range of sizes for the local crops.

    n_local_crops : int
        Number of local crops to create.

    size : int
        The size of the final image.

    Attributes
    ----------
    global_1, global_2 : transforms.Compose
        Two global transforms.

    local : transforms.Compose
        Local transform. Note that the augmentation is stochastic so one
        instance is enough and will lead to different crops.
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1),
        local_crops_scale=(0.05, 0.4),
        n_local_crops=8,
        size=224,
    ):
        self.n_local_crops = n_local_crops
        RandomGaussianBlur = lambda p: transforms.RandomApply(  # noqa
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))],
            p=p,
        )

        flip_and_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        ),
                    ]
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(1.0),  # always apply
                normalize,
            ],
        )

        self.global_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                transforms.RandomSolarize(170, p=0.2),
                normalize,
            ],
        )

        self.local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.5),
                normalize,
            ],
        )

    def __call__(self, img):
        return self.global_1(img), self.local(img)
    

def get_param_groups(model: torch.nn.Module):
    """
    Slightly modified version from:
    https://github.com/facebookresearch/dino/blob/main/utils.py.

    Split model's params into a regularized and not regularized
    group.
    """

    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't regularize biases nor Norm parameters
        if name.endswith('.bias') or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    return [
        {'params': regularized}, 
        {'params': not_regularized, 'weight_decay': 0.}
    ]


def clip_gradients(model, clip):
    norms = []

    for n, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())

            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)

    return np.array(norms)


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    """
    Slightly modified version from:
    https://github.com/facebookresearch/dino/blob/main/utils.py.

    Freeze last layer's parameters to facilite training in first epochs.
    """

    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if 'last_layer' in n:
            p.grad = None


def cosine_scheduler(
        base_val, end_val, n_epochs, iters_per_epoch, warmup_epochs=0, 
        start_warmup_value=0):
    """
    Slightly modified version from:
    https://github.com/facebookresearch/dino/blob/main/utils.py.

    Create an array of learning rates corresponding to a cosine decay schedule 
    with linear warmup.

    Learning rate will start with `start_warmup_value`, increase linearly
    for `warmup_epochs`*`iters_per_epoch` steps up to `base_val`, then
    decay till `end_val` at the end of the training.
    """
    
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * iters_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_val, warmup_iters)

    iters = np.arange(n_epochs * iters_per_epoch - warmup_iters)
    decay_schedule = (
        end_val + 0.5 * (base_val - end_val) * (1 + np.cos(np.pi * iters / len(iters)))
    )

    schedule = np.concatenate((warmup_schedule, decay_schedule))
    assert len(schedule) == n_epochs * iters_per_epoch
    return schedule


def display_gpu_info():
    free_mem, available_mem = torch.cuda.mem_get_info()
    occupied_mem = available_mem - free_mem

    occupied_mem = occupied_mem / 2**30
    available_mem = available_mem / 2**30

    print(f'\nGPU: {occupied_mem:.2f}/{available_mem:.2f} GB occupied\n')


class AverageAggregator():
    """
    Implements a method for calculating mean without storing all the values
    explicitly.
    """

    def __init__(self):
        self._avg = 0
        self._count = 0

    def update(self, val, n=1):
        self._avg = (
            (self._avg*self._count + val*n) / (self._count+n)
        )
        self._count += n

    def item(self):
        return self._avg


if __name__ == '__main__':
    schedules = [
        cosine_scheduler(
            base_val=0.0005,
            end_val=1e-6,
            n_epochs=100,
            iters_per_epoch=100,
            warmup_epochs=10
        ),
        cosine_scheduler(
            base_val=0.04,
            end_val=0.4,
            n_epochs=100,
            iters_per_epoch=100
        ),
        cosine_scheduler(
            base_val=0.9995,
            end_val=1,
            n_epochs=100,
            iters_per_epoch=100
        )
    ]

    for schedule in schedules:
        plt.plot(schedule)
        plt.show()