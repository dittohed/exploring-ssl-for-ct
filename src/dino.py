import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


# TODO
# Doublecheck with
# https://www.youtube.com/watch?v=psmMEWKk4Uk


class Loss(nn.Module):
    """
    Implements loss as described in https://arxiv.org/pdf/2104.14294.pdf.

    Simplified copy-paste from 
    https://github.com/facebookresearch/dino/blob/main/main_dino.py.

    Args:
        ...
    """

    def __init__(self, out_dim, temp_t_warmup, temp_t,
                 temp_t_warmup_epochs, n_epochs, temp_s=0.1,
                 center_momentum=0.9):
        super().__init__()

        self.register_buffer('center', torch.zeros(1, out_dim))

        # Apply warmup for teacher_temp because high temperature makes 
        # the training instable at the beginning
        self.temp_t_schedule = np.concatenate((
            np.linspace(temp_t_warmup, temp_t, 
                        temp_t_warmup_epochs),
            np.ones(n_epochs-temp_t_warmup_epochs) * temp_t
        ))

        self.temp_s = temp_s
        self.center_momentum = center_momentum        

    def forward(self, out_s, out_t, epoch):
        """
        Calculates cross-entropy between softmax outputs of the student and 
        teacher networks.
        """

        soft_s = F.log_softmax(out_s/self.temp_s, dim=-1)
        soft_t = F.softmax(
            (out_t-self.center)/self.temp_t_schedule[epoch], 
            dim=-1
        ).detach()

        loss = torch.sum(-soft_t*soft_s, dim=-1).mean()

        return loss

    @torch.no_grad()
    def update_center(self, batch_center):
        """
        Update center used for teacher output.
        """

        # TODO: doublecheck for multiple GPUs
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(out_t) * dist.get_world_size())
        self.center = (self.center * self.center_momentum 
                      + batch_center * (1 - self.center_momentum))


class Head(nn.Module):
    """
    Implements MLP as described in https://arxiv.org/pdf/2104.14294.pdf.
    Takes aggregated image representation from backbone as input
    (either CLS token for ViT or global average pooling result for Swin).

    Simplified copy-paste from 
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py.

    Args:
        in_dim (int): dimensionality of aggregated image representation.
        out_dim (int): dimensionality of the last layer (softmax is calculated on).
        hidden_dim (int): dimensionality of hidden layers.
        bottleneck_dim (int): dimensionallity of the second to last layer.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # TODO: add layer norm if unstable
     
        layers = [
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        ]
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computes MLP forward pass.

        Args:
            x (torch.Tensor): tensor of shape [n_samples, in_dim].

        Returns:
            torch.Tensor of shape [n_samples, out_dim].
        """

        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x