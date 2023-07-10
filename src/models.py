import torch

from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep


class Backbone(torch.nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.model = SwinTransformer(
            in_chans=1,
            embed_dim=args.embedding_size,
            window_size=ensure_tuple_rep(7, args.spatial_dims),
            patch_size=ensure_tuple_rep(2, args.spatial_dims),
            depths=(2, 2, 2, 2),
            num_heads=(3, 3, 3, 3) if args.low_resource_mode else (3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.drop_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_gradient_checkpointing,
            spatial_dims=args.spatial_dims
        )

    def forward(self, x):
        x = self.model(x)[-1]  # Take the deepest feature map
        b, dim = x.shape[:2]

        # Mean over spatial dimensions
        # TODO: why did the authors use just the first subcube?
        return torch.mean(x.view(b, dim, -1), dim=2)