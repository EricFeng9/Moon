import argparse
import os
from pathlib import Path

from PIL import Image
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch_npu

from common import atomic_json, digest, read
from data import EvaluationSampler, MoonDataset, normalize
from e018_base import metrics
from dinov3_model import build_model, WEIGHTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    rank = int(os.environ['LOCAL_RANK'])
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '8,9,10,11,12,13,14,15'
    device = torch.device(f'npu:{rank}')
    torch.npu.set_device(device)
    torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    dist.init_process_group('hccl')
    checkpoint = torch.load(run/'selection/best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['config']['pretrained_sha256'] == digest(WEIGHTS)
    manifest_path = run/'official_train/manifest.json'
    assert checkpoint['config']['official_manifest_sha256'] == digest(manifest_path)
    assert checkpoint['selected_weights'] == 'model' and checkpoint['best']['epoch'] == checkpoint['epoch']
    model = build_model(checkpoint['config']['variant'])
    model.load_state_dict(checkpoint['model'], strict=True)
    expected = checkpoint['best']
    del checkpoint
    model = model.to(device).eval()
    def loader(records):
        return DataLoader(MoonDataset(records), batch_size=1, sampler=EvaluationSampler(len(records),rank,8),
                          num_workers=2, multiprocessing_context='spawn')
    matrix = torch.zeros((5,5), device=device, dtype=torch.int64)
    with torch.inference_mode():
        for batch in loader(read(manifest_path)['val']):
            image = normalize(batch['image'], device)
            with torch.autocast('npu', dtype=torch.bfloat16):
                pred = model(image).argmax(1)
            labels = batch['mask'].to(device,dtype=torch.long)
            matrix += torch.bincount((labels*5+pred).flatten(), minlength=25).reshape(5,5)
    dist.all_reduce(matrix)
    validation = metrics(matrix.cpu())
    assert abs(validation['mIoU'] - expected['mIoU']) < 1e-6
    root = Path('/home/fengjunming/data/LuSNAR_test')
    records = []
    for number, count in [(2,1597),(4,1320),(9,2087)]:
        scene = f'Moon_{number}'
        paths = sorted((root/scene/'color').glob('*.png'))
        assert len(paths) == count
        (run/'predictions/test'/scene/'submission').mkdir(parents=True, exist_ok=True)
        records.extend(dict(image=str(p),name=p.name,scene=scene) for p in paths)
    predicted = torch.zeros((),device=device,dtype=torch.long)
    with torch.inference_mode():
        for batch in loader(records):
            image = normalize(batch['image'], device)
            votes = torch.zeros((1,5,1024,1024),device=device)
            for size in (768,1024,1280):
                scaled = F.interpolate(image,size=(size,size),mode='bilinear',align_corners=False)
                for flip in (False,True):
                    with torch.autocast('npu',dtype=torch.bfloat16):
                        logits = model(scaled.flip(-1) if flip else scaled)
                    logits = logits.flip(-1) if flip else logits
                    votes += F.interpolate(logits.float(),size=(1024,1024),mode='bilinear',align_corners=False).softmax(1)
            result = votes.argmax(1)[0].to('cpu',torch.uint8).numpy()
            file = run/'predictions/test'/batch['scene'][0]/'submission'/batch['name'][0]
            Image.fromarray(result).save(file)
            predicted += 1
    dist.all_reduce(predicted)
    assert predicted.item() == 5004
    if rank == 0:
        atomic_json(run/'prediction_report.json',dict(best=expected,validation=validation,images=5004,
                    test_source=str(root), predictions=str(run/'predictions'),checkpoint_sha256=digest(run/'selection/best.pt')))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
