import numpy as np
import torch
import matplotlib.pyplot as plt


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