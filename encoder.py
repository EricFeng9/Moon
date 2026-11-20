from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.encoders.mix_transformer import Attention
from torch import nn
from torch.nn import functional as F


class EfficientAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ('q', 'kv', 'attn_drop', 'proj', 'proj_drop', 'sr', 'norm'):
            setattr(self, name, getattr(original, name))
        self.num_heads = original.num_heads
        self.scale = original.scale
        self.sr_ratio = original.sr_ratio

    def forward(self, value, height, width):
        batch, length, channels = value.shape
        query = self.q(value).reshape(batch, length, self.num_heads, channels // self.num_heads)
        query = query.transpose(1, 2).contiguous()
        if self.sr_ratio > 1:
            reduced = self.sr(value.transpose(1, 2).reshape(batch, channels, height, width))
            reduced = self.norm(reduced.flatten(2).transpose(1, 2))
        else:
            reduced = value
        values = self.kv(reduced).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
        key, content = values.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key.contiguous(), content.contiguous(),
                                                 scale=self.scale, dropout_p=self.attn_drop.p if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(attended))


def build_encoder():
    encoder = get_encoder('mit_b2', in_channels=3, depth=5, weights=None)
    for stage in (encoder.block1, encoder.block2, encoder.block3, encoder.block4):
        for block in stage:
            assert isinstance(block.attn, Attention)
            block.attn = EfficientAttention(block.attn)
    return encoder
