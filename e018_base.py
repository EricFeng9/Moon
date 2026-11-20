import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

from encoder import build_encoder


def conv_norm(inputs, outputs, kernel=3):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(16, outputs), nn.GELU())


class PrototypeContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.presence = nn.Linear(192, 5)
        self.update = nn.Linear(384, 1)
        self.memory_norm = nn.LayerNorm(192)
        self.query = nn.Conv2d(192, 96, 1)
        self.key = nn.Linear(192, 96)
        self.value = nn.Linear(192, 192)
        self.fuse = conv_norm(384, 192, 1)

    def forward(self, feature, guide, previous, embedding):
        batch, channels, height, width = feature.shape
        probabilities = guide.float().flatten(2).softmax(1)
        weights = probabilities / probabilities.sum(2, keepdim=True).clamp_min(1e-6)
        regions = torch.bmm(weights, feature.float().flatten(2).transpose(1, 2))
        previous = embedding.unsqueeze(0).expand(batch, -1, -1) if previous is None else previous
        gate = self.update(torch.cat((regions, previous), dim=2)).float().sigmoid()
        memory = self.memory_norm(gate * regions + (1.0 - gate) * previous + embedding.unsqueeze(0))
        query = F.normalize(self.query(feature).float().flatten(2), dim=1, eps=1e-6)
        key = F.normalize(self.key(memory).float(), dim=2, eps=1e-6)
        affinity = 10.0 * torch.bmm(key, query)
        presence = self.presence(feature.mean((2, 3)))
        attention = (affinity + presence.float().sigmoid().clamp_min(1e-4).log().unsqueeze(2)).softmax(1)
        context = torch.bmm(self.value(memory).float().transpose(1, 2), attention)
        context = context.reshape(batch, channels, height, width).to(feature.dtype)
        return feature + self.fuse(torch.cat((feature, context), dim=1)), memory, affinity.reshape(batch, 5, height, width), presence


class PyramidDecoder(nn.Module):
    def __init__(self, prototype):
        super().__init__()
        self.prototype = prototype
        self.lateral = nn.ModuleList([conv_norm(channels, 192, 1) for channels in (512, 320, 128, 64)])
        self.refine = nn.ModuleList([nn.Sequential(conv_norm(192, 192), conv_norm(192, 192)) for _ in range(4)])
        self.classifiers = nn.ModuleList([nn.Conv2d(192, 5, 1) for _ in range(4)])
        if prototype:
            self.context = nn.ModuleList([PrototypeContext() for _ in range(4)])
            self.class_embedding = nn.Parameter(torch.randn(5, 192) * 0.02)

    def forward(self, features):
        previous, memory = None, None
        logits, affinities, presence, decoded = [], [], [], []
        for index, feature in enumerate(features[2:][::-1]):
            current = self.lateral[index](feature)
            if previous is not None:
                current = current + F.interpolate(previous, size=current.shape[-2:], mode='bilinear', align_corners=False)
            current = self.refine[index](current)
            guide = self.classifiers[index](current)
            if self.prototype:
                current, memory, affinity, exists = self.context[index](current, guide, memory, self.class_embedding)
                affinities.append(affinity)
                presence.append(exists)
                guide = self.classifiers[index](current)
            logits.append(guide)
            decoded.append(current)
            previous = current
        return logits, affinities, presence, decoded


class HierarchyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sky = nn.Sequential(conv_norm(192, 96), conv_norm(96, 64), nn.Conv2d(64, 1, 1))
        self.mountain = nn.Sequential(conv_norm(192, 96), conv_norm(96, 64), nn.Conv2d(64, 1, 1))
        self.surface = nn.Sequential(conv_norm(192, 96), conv_norm(96, 64), nn.Conv2d(64, 3, 1))
        self.fusion_gate = nn.Conv2d(192, 1, 1)

    def forward(self, decoded, flat):
        size = decoded[-1].shape[-2:]
        sky = F.interpolate(self.sky(decoded[1]).float(), size=size, mode='bilinear', align_corners=False)
        mountain = F.interpolate(self.mountain(decoded[2]).float(), size=size, mode='bilinear', align_corners=False)
        surface = self.surface(decoded[3]).float().log_softmax(1)
        ground = F.logsigmoid(-sky)
        near = ground + F.logsigmoid(-mountain)
        hierarchical = torch.cat((near + surface, ground + F.logsigmoid(mountain), F.logsigmoid(sky)), dim=1)
        gate = self.fusion_gate(decoded[-1]).float().sigmoid()
        combined = (1.0 - gate) * flat.float().log_softmax(1) + gate * hierarchical
        return combined, hierarchical


class Architecture(nn.Module):
    def __init__(self, specification):
        super().__init__()
        assert set(specification) == {'prototype', 'hierarchy', 'illumination'}
        self.specification = dict(specification)
        self.encoder = build_encoder()
        self.decoder = PyramidDecoder(specification['prototype'])
        if specification['hierarchy']:
            self.hierarchy = HierarchyHead()
        if specification['illumination']:
            self.view_fusion = nn.ModuleList([conv_norm(channels * 2, channels, 1) for channels in (64, 128, 320, 512)])
            self.view_classifier = nn.Conv2d(64, 5, 1)

    @staticmethod
    def illumination_view(image):
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        rgb = image.float() * std + mean
        local = F.avg_pool2d(rgb, 31, 1, 15, count_include_pad=False)
        contrast = rgb - local
        scale = (contrast.square().mean((2, 3), keepdim=True) + 1e-4).sqrt()
        transformed = 0.5 + 0.2 * torch.tanh(contrast / scale)
        return (transformed - mean) / std

    def encode_views(self, image):
        features = self.encoder(image)
        views = []
        if self.specification['illumination']:
            second = self.encoder(self.illumination_view(image))
            views = [self.view_classifier(features[2]), self.view_classifier(second[2])]
            features = features[:2] + [layer(torch.cat((first, alternate), dim=1))
                                      for layer, first, alternate in zip(self.view_fusion, features[2:], second[2:])]
        return features, views

    def render_logits(self, image, decoded, logits):
        return F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, image):
        features, views = self.encode_views(image)
        logits, affinity, presence, decoded = self.decoder(features)
        expert = None
        final = logits[-1]
        if self.specification['hierarchy']:
            final, expert = self.hierarchy(decoded, final)
        main = self.render_logits(image, decoded, final)
        if self.training:
            return dict(logits=main, auxiliary=logits[:-1], affinity=affinity, presence=presence, expert=expert, views=views)
        return main


def build_model(specification):
    return Architecture(specification)


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True, smooth=1e-6, ignore_index=255)

    def base(self, logits, target):
        logits = logits.float()
        return F.cross_entropy(logits, target, ignore_index=255) + self.dice(logits, target)

    @staticmethod
    def resize_target(target, logits):
        return F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:], mode='nearest')[:, 0].long()

    def forward(self, output, target):
        loss = self.base(output['logits'], target)
        loss = loss + 0.4 * torch.stack([self.base(logits, self.resize_target(target, logits))
                                        for logits in output['auxiliary']]).mean()
        if output['affinity']:
            balanced = []
            classes = torch.arange(5, device=target.device).reshape(1, 5, 1, 1)
            for affinity in output['affinity']:
                current = self.resize_target(target, affinity)
                pixels = F.cross_entropy(affinity.float(), current, ignore_index=255, reduction='none')
                members = current.unsqueeze(1) == classes
                counts = members.sum((0, 2, 3))
                total = (members * pixels.unsqueeze(1)).sum((0, 2, 3))
                valid = counts > 0
                balanced.append(((total / counts.clamp_min(1)) * valid).sum() / valid.sum())
            exists = (target.unsqueeze(1) == classes).flatten(2).any(2).float()
            presence = torch.stack([F.binary_cross_entropy_with_logits(value.float(), exists)
                                     for value in output['presence']]).mean()
            loss = loss + 0.1 * torch.stack(balanced).mean() + 0.05 * presence
        if output['expert'] is not None:
            loss = loss + 0.2 * self.base(output['expert'], self.resize_target(target, output['expert']))
        if output['views']:
            first, second = [value.float() for value in output['views']]
            current = self.resize_target(target, first)
            loss = loss + 0.2 * (self.base(first, current) + self.base(second, current)) / 2.0
            log_first, log_second = first.log_softmax(1), second.log_softmax(1)
            divergence = 0.5 * ((log_first.exp() - log_second.exp()) * (log_first - log_second)).sum(1)
            valid = current != 255
            loss = loss + 0.05 * (divergence * valid).sum() / valid.sum()
        return loss


def metrics(matrix):
    matrix = matrix.double()
    true_positive = matrix.diag()
    support = matrix.sum(1)
    union = support + matrix.sum(0) - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.zeros_like(union))
    return dict(mIoU=iou[support > 0].mean().item(), IoU=iou.tolist(),
                support=support.long().tolist(), confusion_matrix=matrix.long().tolist())
