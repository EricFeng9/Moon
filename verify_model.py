import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, digest, read
from data import MoonDataset, normalize
from dinov3_model import INITIAL_DETAIL_FRACTION, WEIGHTS, build_model
from e018_base import SegmentationLoss


def verify(run):
    torch.set_num_threads(4)
    manifest = read(run/'official_train/manifest.json')
    row = MoonDataset(manifest['train'])[0]
    image = normalize(row['image'][None], torch.device('cpu'))
    target = row['mask'][None].long()
    dependency = read(ROOT/'dependencies.json')
    assert digest(WEIGHTS) == dependency['sha256']
    torch.manual_seed(42)
    model = build_model('dinov3_subpixel_e018').eval()
    with torch.no_grad():
        pyramid, _ = model.encode_views(image)
        shapes = [list(value.shape) for value in pyramid[2:]]
        assert shapes == [[1,64,256,256],[1,128,128,128],[1,320,64,64],[1,512,32,32]]
        logits = model(image)
        assert logits.shape == (1,5,1024,1024) and logits.isfinite().all()
        torch.manual_seed(42)
        parent = build_model('dinov3_gate_e018').eval()
        torch.testing.assert_close(logits, parent(image), rtol=0, atol=0)
    torch.manual_seed(42)
    model.train()
    assert not model.encoder.dino.training
    assert not any(p.requires_grad for p in model.encoder.dino.parameters())
    frozen = model.encoder.dino.patch_embed.proj.weight
    frozen_before = frozen.detach().clone()
    inputs = F.interpolate(image,size=(256,256),mode='bilinear',align_corners=False)
    mask = F.interpolate(target[:,None].float(),size=(256,256),mode='nearest')[:,0].long()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    maps = []

    def record_gate(_gate, _inputs, output):
        maps.append(output.detach().sigmoid())

    hooks = [gate.register_forward_hook(record_gate) for gate in model.encoder.detail_gates]
    loss = SegmentationLoss()(model(inputs),mask)
    assert [tuple(value.shape) for value in maps] == [(1,1,64,64),(1,1,32,32)]
    for value in maps:
        torch.testing.assert_close(value, torch.full_like(value, INITIAL_DETAIL_FRACTION))
    loss.backward()
    trainable = {k:p for k,p in model.named_parameters() if p.requires_grad}
    for name, parameter in trainable.items():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
    branches = []
    for gate in model.encoder.detail_gates:
        semantic, detail = gate.weight.grad.chunk(2, dim=1)
        norms = dict(dino=semantic.norm().item(), mit=detail.norm().item())
        assert min(norms.values()) > 0
        branches.append(norms)
    assert all(p.grad is None for p in model.encoder.dino.parameters())
    assert model.upsampler.head.weight.grad.norm().item() > 0
    optimizer.step()
    assert torch.equal(frozen, frozen_before)
    maps.clear()
    with torch.no_grad():
        model.encode_views(inputs)
    for hook in hooks:
        hook.remove()
    ranges = [(value.min().item(),value.max().item()) for value in maps]
    assert len(ranges) == 2 and all(0 < low < high < 1 for low,high in ranges)
    with torch.no_grad():
        after = model.eval()(image)
        assert after.isfinite().all() and not torch.equal(after, logits)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    SegmentationLoss()(model(inputs), mask).backward()
    subpixel_gradient = model.upsampler.project[0].weight.grad.norm().item()
    assert subpixel_gradient > 0
    atomic_json(run/'parent_verification.json',dict(sample=row['name'],pyramid=shapes,loss=float(loss.detach()),
                frozen_parameters=sum(p.numel() for p in model.encoder.dino.parameters()),
                trainable_parameters=sum(p.numel() for p in trainable.values()),
                gate_initial_fraction=INITIAL_DETAIL_FRACTION,gate_branch_gradient_norms=branches,
                gate_ranges_after_one_update=ranges,subpixel_input_gradient=subpixel_gradient,
                output_identical_at_initialization=True,
                frozen_weights_unchanged_after_step=True,sha256=dependency['sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
