from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import tv_tensors
from torchvision.transforms import v2


CLASS_NAMES = ('regolith', 'rock', 'crater', 'mountain', 'sky')
PALETTE = np.array([[187, 70, 156], [232, 250, 80], [120, 0, 200],
                    [173, 69, 31], [34, 201, 248]], dtype=np.uint8)
SOURCE_TO_COMPETITION = np.array([0, 2, 1, 3, 4], dtype=np.uint8)
AUGMENTATION = dict(resize_min=512, resize_max_exclusive=2049, crop_size=1024,
                    flip_probability=0.5, brightness=[0.875, 1.125], contrast=[0.5, 1.5],
                    saturation=[0.5, 1.5], hue=[-0.05, 0.05], photometric_probability=0.5,
                    image_padding=0, mask_padding=255)


class MoonDataset(Dataset):
    def __init__(self, records, training=False, seed=42):
        self.records = records
        self.training = training
        self.seed = seed
        self.augmentation = v2.Compose([
            v2.RandomResize(512, 2049, antialias=True),
            v2.RandomCrop((1024, 1024), pad_if_needed=True,
                          fill={tv_tensors.Image: 0, tv_tensors.Mask: 255}),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPhotometricDistort(brightness=(0.875, 1.125), contrast=(0.5, 1.5),
                                       saturation=(0.5, 1.5), hue=(-0.05, 0.05), p=0.5),
        ]) if training else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        index, epoch = item if isinstance(item, tuple) else (item, 0)
        record = self.records[index]
        with Image.open(record['image']) as source:
            image = source.convert('RGB')
        mask = None
        if 'mask' in record:
            with Image.open(record['mask']) as source:
                assert source.mode == 'P' and source.getpalette()[:15] == [187,70,156,120,0,200,232,250,80,173,69,31,34,201,248]
                mask = Image.fromarray(SOURCE_TO_COMPETITION[np.asarray(source)])
        assert image.size == (1024, 1024)
        image = v2.functional.to_image(image)
        if mask is not None:
            mask = tv_tensors.Mask(np.asarray(mask).copy())
        if self.training:
            with torch.random.fork_rng(devices=[], device_type='cpu'):
                torch.random.default_generator.manual_seed(self.seed + epoch * 1_000_003 + index * 101)
                image, mask = self.augmentation(image, mask)
        result = dict(image=image.as_subclass(torch.Tensor),
                      index=index, name=record['name'], scene=record['scene'])
        if mask is not None:
            result['mask'] = mask.as_subclass(torch.Tensor)
        return result


class EpochSampler(Sampler):
    def __init__(self, dataset, rank, world_size, seed=42):
        assert world_size in (8, 16) and 0 <= rank < world_size
        # 保留原16份数据排列，每个新进程交错读取其中两份。
        self.samplers = [DistributedSampler(dataset, num_replicas=16, rank=rank + branch * world_size,
                                            shuffle=True, seed=seed, drop_last=False)
                         for branch in range(16 // world_size)]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        for sampler in self.samplers:
            sampler.set_epoch(epoch)

    def __iter__(self):
        return iter((index, self.epoch) for group in zip(*(iter(sampler) for sampler in self.samplers))
                    for index in group)

    def __len__(self):
        return sum(len(sampler) for sampler in self.samplers)


class EvaluationSampler(Sampler):
    def __init__(self, size, rank, world_size):
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def normalize(image, device):
    image = image.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return image.sub_(mean).div_(std)
