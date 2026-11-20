import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from context import EvidenceDecoder
from e018_base import Architecture


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor'
WEIGHTS = ROOT / 'weights/dinov3_vits16_pretrain_lvd1689m-08c60483.bin'
VARIANTS = ('dinov3_subpixel_e018',)
INITIAL_DETAIL_FRACTION = 0.05


class ResidualEncoder(nn.Module):
    def __init__(self, mit):
        super().__init__()
        self.patch_embed1, self.block1, self.norm1 = mit.patch_embed1, mit.block1, mit.norm1
        self.patch_embed2, self.block2, self.norm2 = mit.patch_embed2, mit.block2, mit.norm2
        self.dino = torch.hub.load(str(VENDOR), 'dinov3_vits16', source='local', pretrained=False)
        self.dino.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
        self.dino.requires_grad_(False).eval()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(384, channels, 1, bias=False), nn.GroupNorm(16, channels))
            for channels in (64, 128, 320, 512)
        ])
        self.detail_norms = nn.ModuleList([nn.GroupNorm(16, 64), nn.GroupNorm(16, 128)])
        self.detail_gates = nn.ModuleList([nn.Conv2d(2 * channels, 1, 1) for channels in (64, 128)])
        initial_logit = math.log(INITIAL_DETAIL_FRACTION / (1 - INITIAL_DETAIL_FRACTION))
        for gate in self.detail_gates:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, initial_logit)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, image, contrast):
        with torch.no_grad():
            tokens = self.dino.get_intermediate_layers(image, n=[2, 5, 8, 11], reshape=True)
        detail4 = self.norm1(self.block1(self.patch_embed1(contrast))).contiguous()
        detail8 = self.norm2(self.block2(self.patch_embed2(detail4))).contiguous()
        semantics = [projection(feat) for projection, feat in zip(self.projections, tokens)]
        features = [
            F.interpolate(semantics[i], size=detail.shape[-2:], mode='bilinear', align_corners=False)
            for i, detail in enumerate((detail4, detail8))
        ]
        primary_view = features[0]
        for i, detail in enumerate((detail4, detail8)):
            semantic = features[i]
            detail = self.detail_norms[i](detail)
            logits = self.detail_gates[i](torch.cat((semantic, detail), dim=1))
            fraction = logits.float().sigmoid().to(detail.dtype)
            features[i] = semantic + fraction * detail
        features.append(semantics[2])
        features.append(F.avg_pool2d(semantics[3], 2, 2, ceil_mode=True))
        return [image, None, *features], (primary_view, detail4)


class DinoResidualSegmentation(Architecture):
    def __init__(self, parent):
        nn.Module.__init__(self)
        self.specification = dict(parent.specification)
        self.encoder = ResidualEncoder(parent.encoder)
        self.decoder = parent.decoder
        self.hierarchy = parent.hierarchy
        self.view_classifier = parent.view_classifier

    def encode_views(self, image):
        features, views = self.encoder(image, self.illumination_view(image))
        return features, [self.view_classifier(view) for view in views] if self.training else []


class SubpixelResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(192 + 3*16, 32, 1, bias=False),
                                     nn.GroupNorm(8, 32), nn.GELU())
        self.spatial = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.head = nn.Conv2d(32, 5*16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image, decoded):
        height, width = image.shape[-2:]
        padded_h, padded_w = (4 * size for size in decoded.shape[-2:])
        pad = (0, padded_w-width, 0, padded_h-height)
        rgb = F.pad(image, pad, mode='replicate') if pad[1] or pad[3] else image
        local_rgb = F.pixel_unshuffle(rgb.to(decoded.dtype), 4)
        feature = self.project(torch.cat((decoded, local_rgb), dim=1))
        offsets = self.head(F.gelu(feature + self.spatial(feature))).float()
        offsets = offsets.reshape(image.shape[0], 5, 16, padded_h//4, padded_w//4)
        offsets = offsets - offsets.mean(dim=2, keepdim=True)
        return F.pixel_shuffle(offsets.flatten(1,2), 4)[..., :height, :width]


class DinoSubpixelSegmentation(DinoResidualSegmentation):
    def __init__(self, parent):
        super().__init__(parent)
        self.upsampler = SubpixelResidual()

    def render_logits(self, image, decoded, logits):
        return super().render_logits(image, decoded, logits) + self.upsampler(image, decoded[-1])


def build_model(variant):
    assert variant in ('baseline', 'dinov3_gate_e018') or variant in VARIANTS
    model = Architecture(dict(prototype=True, hierarchy=True, illumination=True))
    model.decoder = EvidenceDecoder(model.decoder, 'region_evidence')
    if variant == 'baseline':
        return model
    if variant == 'dinov3_gate_e018':
        return DinoResidualSegmentation(model)
    return DinoSubpixelSegmentation(model)
