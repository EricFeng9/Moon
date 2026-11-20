import argparse
from contextlib import nullcontext
import gc
import os
from pathlib import Path
from statistics import median
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch_npu

from common import atomic_json, read
from data import MoonDataset, normalize
from e018_base import SegmentationLoss
from dinov3_model import build_model


def timed_blocks(variant, batch, device):
    torch.manual_seed(42)
    model = build_model(variant)
    parameters = sum(p.numel() for p in model.parameters())
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False,
                                    gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=3e-4)
    criterion = SegmentationLoss().to(device)
    image,target = batch
    def update():
        optimizer.zero_grad(set_to_none=True)
        for i in range(2):
            with model.no_sync() if i==0 else nullcontext():
                with torch.autocast('npu',dtype=torch.bfloat16):
                    loss = criterion(model(image[i::2]),target[i::2])
                (loss*0.5).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
    def predict(six):
        original = image[:1]
        with torch.inference_mode(), torch.autocast('npu',dtype=torch.bfloat16):
            if not six:
                return model.module(original).argmax(1)
            votes = torch.zeros((1,5,1024,1024),device=device)
            for size in (768,1024,1280):
                scaled = F.interpolate(original,size=(size,size),mode='bilinear',align_corners=False)
                for flip in (False,True):
                    logits = model.module(scaled.flip(-1) if flip else scaled)
                    logits = logits.flip(-1) if flip else logits
                    votes += F.interpolate(logits.float(),size=(1024,1024),mode='bilinear',align_corners=False).softmax(1)
            return votes.argmax(1)
    result = dict(parameters=parameters)
    for mode,operation,repeats,inner in [('train',update,6,1),('single',lambda:predict(False),8,4),
                                         ('six',lambda:predict(True),5,1)]:
        model.train(mode=='train')
        for _ in range(3):
            operation()
        torch.npu.synchronize()
        times = []
        for _ in range(repeats):
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(inner):
                operation()
            torch.npu.synchronize()
            elapsed = torch.tensor((time.perf_counter()-start)/inner,device=device)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            times.append(elapsed.item())
        result[mode] = times
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    run=parser.parse_args().run
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='8,9,10,11,12,13,14,15'
    rank=int(os.environ['LOCAL_RANK'])
    device=torch.device(f'npu:{rank}')
    torch.npu.set_device(device)
    torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
    torch.npu.matmul.allow_hf32=False
    torch.npu.conv.allow_hf32=False
    dist.init_process_group('hccl')
    dataset=MoonDataset(read(run/'official_train/manifest.json')['train'],training=True)
    records=[dataset[(113*j+rank,0)] for j in range(4)]
    batch=(normalize(torch.stack([row['image'] for row in records]),device),
           torch.stack([row['mask'] for row in records]).to(device,dtype=torch.long))
    report=dict(measurement='max rank wall time, median of warmed repeats; resident training images; eight devices',
                scope='training update global batch 32; inference batch 1 per device; model computation and six-view aggregation',
                max_time_ratio=1.10, runs=[])
    for variant in [read(run/'parent.json')['variant']]:
        pair=[]
        for architecture in ('baseline',variant,variant,'baseline'):
            result=timed_blocks(architecture,batch,device)
            pair.append(result)
            report['runs'].append(dict(architecture=architecture,**result))
            if rank==0:
                atomic_json(run/'benchmark_progress.json',report)
            gc.collect()
            torch.npu.empty_cache()
        timings={name:{mode:median([median(pair[i][mode]) for i in slots]) for mode in ('train','single','six')}
                 for name,slots in [('baseline',(0,3)),(variant,(1,2))]}
        ratios={mode:timings[variant][mode]/timings['baseline'][mode] for mode in timings[variant]}
        report[variant]=dict(seconds=timings,ratios=ratios,parameters=pair[1]['parameters'])
        report['within_speed_target']=max(ratios.values()) <= report['max_time_ratio']
        report['selected']=variant
    if rank==0:
        atomic_json(run/'benchmark.json',report)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
