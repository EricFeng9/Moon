import math

import torch
from torch import nn
from torch.nn import functional as F

from e018_base import conv_norm


def box_mean(value, kernel):
    padding = kernel // 2
    horizontal = F.avg_pool2d(value.float(), (1, kernel), 1, (0, padding), count_include_pad=False)
    return F.avg_pool2d(horizontal, (kernel, 1), 1, (padding, 0), count_include_pad=False)


def box_sum(value, kernel):
    padding = kernel // 2
    horizontal = F.avg_pool2d(value.float(), (1, kernel), 1, (0, padding), count_include_pad=True)
    return F.avg_pool2d(horizontal, (kernel, 1), 1, (padding, 0), count_include_pad=True) * kernel ** 2


def ring_evidence(value):
    valid = torch.ones_like(value[:, :1], dtype=torch.float32)
    totals = [box_sum(value, kernel) for kernel in (3, 7, 15)]
    counts = [box_sum(valid, kernel) for kernel in (3, 7, 15)]
    center = totals[0] / counts[0]
    near = (totals[1] - totals[0]) / (counts[1] - counts[0])
    far = (totals[2] - totals[1]) / (counts[2] - counts[1])
    return value.float() - center, center - near, near - far


def semantic_evidence(guide, coarse_guide):
    probabilities = guide.float().softmax(1)
    coarse = F.interpolate(coarse_guide.float(), size=guide.shape[-2:], mode='bilinear',
                           align_corners=False).softmax(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(1, keepdim=True) / math.log(5)
    disagreement = (probabilities - coarse).abs().mean(1, keepdim=True)
    return probabilities, torch.cat((probabilities, entropy, disagreement), dim=1)


class RingEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = conv_norm(192, 48, 1)
        self.semantic = conv_norm(192, 48, 1)
        self.selection = nn.Sequential(conv_norm(55, 48, 1), nn.Conv2d(48, 3, 1))
        self.fuse = nn.Sequential(conv_norm(144, 96, 1),
                                  nn.Conv2d(96, 96, 3, padding=1, groups=96, bias=False),
                                  nn.GroupNorm(16, 96), nn.GELU(), conv_norm(96, 192, 1))
        self.gain = nn.Parameter(torch.tensor(-1.0))

    def forward(self, feature, guide, coarse, coarse_guide):
        projected = self.project(feature)
        probabilities, evidence = semantic_evidence(guide, coarse_guide)
        selection = self.selection(torch.cat((projected.float(), evidence), dim=1).to(feature.dtype)).float().softmax(1)
        rings = ring_evidence(projected)
        contrast = sum(selection[:, index:index + 1] * value for index, value in enumerate(rings))
        semantic = F.interpolate(self.semantic(coarse).float(), size=feature.shape[-2:], mode='bilinear',
                                 align_corners=False)
        context = torch.cat((projected.float(), contrast, semantic), dim=1).to(feature.dtype)
        surface = probabilities[:, :3].sum(1, keepdim=True)
        correction = self.gain.sigmoid() * surface * self.fuse(context).float()
        return feature + correction.to(feature.dtype)


class RegionEvidence(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.project = conv_norm(192, 48, 1)
        self.semantic = conv_norm(192, 48, 1)
        self.fuse = nn.Sequential(conv_norm(240, 96, 1),
                                  nn.Conv2d(96, 96, 3, padding=1, groups=96, bias=False),
                                  nn.GroupNorm(16, 96), nn.GELU(), conv_norm(96, 192, 1))
        self.gate = nn.Sequential(conv_norm(55, 48, 1), nn.Conv2d(48, 1, 1))
        self.gain = nn.Parameter(torch.tensor(-1.0))

    def forward(self, feature, guide, coarse, coarse_guide):
        projected = self.project(feature).float()
        probabilities, evidence = semantic_evidence(guide, coarse_guide)
        surface = probabilities[:, :3]
        batch, _, height, width = projected.shape
        mass = box_mean(surface, self.kernel)
        weighted = surface.unsqueeze(2) * projected.unsqueeze(1)
        numerator = box_mean(weighted.reshape(batch, 144, height, width), self.kernel)
        numerator = numerator.reshape(batch, 3, 48, height, width)
        positive = numerator / mass.unsqueeze(2).clamp_min(1e-4)
        other_mass = mass.sum(1, keepdim=True) - mass
        negative = (numerator.sum(1, keepdim=True) - numerator) / other_mass.unsqueeze(2).clamp_min(1e-4)
        contrast = (positive - negative).reshape(batch, 144, height, width)
        semantic = F.interpolate(self.semantic(coarse).float(), size=(height, width), mode='bilinear',
                                 align_corners=False)
        context = torch.cat((projected, contrast, semantic), dim=1).to(feature.dtype)
        gate = self.gate(torch.cat((projected, evidence), dim=1).to(feature.dtype)).float().sigmoid()
        correction = self.gain.sigmoid() * (0.25 + 0.75 * gate) * surface.sum(1, keepdim=True) * self.fuse(context).float()
        return feature + correction.to(feature.dtype)


class EvidenceDecoder(nn.Module):
    def __init__(self, original, extension):
        super().__init__()
        self.original = original
        self.extension = extension
        assert extension in ('ring_evidence', 'region_evidence')
        self.refiners = nn.ModuleList([RingEvidence(), RingEvidence()] if extension == 'ring_evidence'
                                      else [RegionEvidence(9), RegionEvidence(17)])

    @property
    def lateral(self):
        return self.original.lateral

    @property
    def classifiers(self):
        return self.original.classifiers

    def forward(self, features):
        logits, affinities, presence, decoded = self.original(features)
        for index, refiner in zip((2, 3), self.refiners):
            current = refiner(decoded[index], logits[index], decoded[0], logits[0])
            decoded[index] = current
            logits[index] = self.classifiers[index](current)
        return logits, affinities, presence, decoded
