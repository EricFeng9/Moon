import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch_npu

from data import AUGMENTATION, CLASS_NAMES, EpochSampler, EvaluationSampler, MoonDataset, normalize
from common import digest
from e018_base import SegmentationLoss, metrics
from dinov3_model import build_model, VARIANTS, WEIGHTS
from epoch_policy import validate_resume


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger('moon1115')


def atomic_json(path, value):
    pending = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def publish(checkpoint, output, improved):
    pending = output / f'.checkpoint_{uuid.uuid4().hex}.pending'
    with pending.open('xb') as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    latest = output / 'latest.pt'
    os.replace(pending, latest)
    targets = ['best.pt'] if improved else []
    for name in targets:
        target = output / name
        staged = output / f'.{name}.{uuid.uuid4().hex}.pending'
        staged.hardlink_to(latest)
        os.replace(staged, target)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                npu=torch.npu.get_rng_state(device).cpu())


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.npu.set_rng_state(state['npu'].cpu(), device)


def loader(dataset, sampler, batch_size, workers):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      multiprocessing_context='spawn' if workers else None)


def confusion(logits, target):
    return torch.bincount((target * 5 + logits.argmax(1)).reshape(-1), minlength=1280)[:25].reshape(5, 5)


def evaluate(model, data, device, maximum=0):
    model.eval()
    matrix = torch.zeros((5, 5), dtype=torch.int64, device=device)
    samples = torch.zeros((), dtype=torch.int64, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(data):
            if maximum and index >= maximum:
                break
            image = normalize(batch['image'], device)
            target = batch['mask'].to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast('npu', dtype=torch.bfloat16):
                logits = model(image)
            matrix += confusion(logits, target)
            samples += len(target)
    dist.all_reduce(matrix)
    dist.all_reduce(samples)
    return dict(metrics(matrix.cpu()), samples=samples.item(), seconds=time.perf_counter() - started)


def status(output, rank, phase, epoch, step, steps, history, best, started, **extra):
    if rank:
        return
    recent = history[-3:]
    completed = history[-1]['epoch'] if history else 0
    elapsed = time.perf_counter() - started
    atomic_json(output / 'status.json', dict(phase=phase, epoch=epoch, step=step, steps=steps,
                completed_epoch=completed, elapsed_seconds=elapsed, best=best,
                latest_validation=history[-1].get('validation') if history else None,
                epoch_seconds=sum(row['seconds'] for row in recent) / len(recent) if recent else None,
                pid=os.getpid(), torchrun_pid=os.getppid(), screen=os.environ.get('STY'), **extra))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--schedule-epochs', type=int, default=0)
    parser.add_argument('--mode', choices=('probe', 'selection'), required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--probe-steps', type=int, default=100)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--disable-early-stopping', action='store_true')
    parser.add_argument('--extend-epochs', action='store_true')
    parser.add_argument('--redistribute-devices', action='store_true')
    parser.add_argument('--budget-seconds', type=int, default=36000)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--variant', choices=tuple(VARIANTS), required=True)
    parser.add_argument('--training-deadline', type=float)
    parser.add_argument('--unlimited-runtime', action='store_true')
    parser.add_argument('--epoch-cost-seconds', type=float, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--encoder-lr', type=float, default=3e-4)
    parser.add_argument('--head-lr', type=float, default=6e-4)
    args = parser.parse_args()
    if not args.unlimited_runtime and args.training_deadline is None:
        parser.error('--training-deadline is required unless --unlimited-runtime is set')
    assert args.epochs > 0 and args.batch_size > 0 and args.workers >= 0 and args.patience >= 0
    assert args.disable_early_stopping == (args.patience == 0)
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    physical_devices = [int(value) for value in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')]
    assert world_size == 8 and physical_devices == list(range(8, 16)) and rank == local_rank
    assert args.batch_size == 4
    device = torch.device(f'npu:{local_rank}')
    torch.npu.set_device(device)
    torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    dist.init_process_group('hccl', timeout=timedelta(minutes=30))
    output = args.output_dir.resolve()
    if rank == 0:
        if output.exists() and args.resume is None:
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    attempts = [None]
    if rank == 0:
        previous = list(output.glob('attempt_*'))
        attempts[0] = f'attempt_{len(previous) + 1:04d}'
        (output / attempts[0]).mkdir()
    dist.broadcast_object_list(attempts)
    attempt = output / attempts[0]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(attempt / f'rank{rank}.log')])

    def report_exception(kind, value, trace):
        LOGGER.critical('训练异常', exc_info=(kind, value, trace))

    sys.excepthook = report_exception
    source_files = ('data.py', 'e018_base.py', 'context.py', 'encoder.py', 'train.py', 'dinov3_model.py',
                    'predict.py', 'run_npu.sh', 'common.py', 'epoch_policy.py', 'benchmark.py', 'run.py',
                    'dependencies.json')
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    if rank == 0:
        snapshot = attempt / 'source'
        snapshot.mkdir()
        for name in source_files:
            shutil.copy2(ROOT / name, snapshot / name)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    manifest = json.loads(args.manifest.read_text())
    assert manifest['source'] == '/home/fengjunming/data/LuSNAR_train.rar'
    for rows, scenes in [(manifest['train'], {1,3,6,7,8}), (manifest['val'], {5})]:
        assert {row['scene'] for row in rows} == {f'Moon_{i}' for i in scenes}
        for row in rows:
            for key, folder in [('image','color'), ('mask','label')]:
                assert Path(row[key]).resolve() == args.manifest.resolve().parent / row['scene'] / folder / row['name']
    records = manifest['train']
    training = MoonDataset(records, training=True)
    validation = MoonDataset(manifest['val'])
    sampler = EpochSampler(training, rank, world_size)
    train_loader = loader(training, sampler, args.batch_size, args.workers)
    val_loader = loader(validation, EvaluationSampler(len(validation), rank, world_size), 1, args.workers)
    model = build_model(args.variant)
    initial_digests = dict(
        encoder=hashlib.sha256(model.encoder.patch_embed1.proj.weight.detach().cpu().numpy().tobytes()).hexdigest(),
        decoder=hashlib.sha256(model.decoder.lateral[0][0].weight.detach().cpu().numpy().tobytes()).hexdigest())
    assert initial_digests == dict(encoder='16c637615a04bfcc7f674e20e69e98b927d76e884c2116b5e4c31355cf5bdc76',
                                   decoder='0cc1f1b2bd1a93cc6bfb9e1d34822a40a0b4d73dbd4963829faf695426cd4894')
    initial_gate_weights = [gate.weight.detach().clone() for gate in model.encoder.detail_gates]
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        encoder = name.startswith('encoder.')
        decay = parameter.ndim > 1 and not name.endswith('.bias')
        groups.setdefault((encoder, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=values, lr=args.encoder_lr if encoder else args.head_lr,
                                       weight_decay=0.01 if decay else 0.0)
                                  for (encoder, decay), values in groups.items()],
                                 betas=(0.9, 0.999), eps=1e-8)
    schedule_epochs = args.schedule_epochs or args.epochs
    assert schedule_epochs >= args.epochs
    total_steps = schedule_epochs * len(train_loader)
    warmup_steps = min(3 * len(train_loader), max(1, total_steps // 10))

    def schedule(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return max(0.01, (1.0 - progress) ** 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                    gradient_as_bucket_view=True)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.npu.manual_seed(seed + rank)
    config = dict(model='E018_region', architecture='frozen_dinov3_gated_subpixel_logits', variant=args.variant,
                  initialization='frozen_dinov3_pretrained_other_parameters_random',
                  pretrained_sha256=digest(WEIGHTS),
                  initial_checkpoint=None, random_initialization_digests=initial_digests, patience=args.patience,
                  early_stopping_enabled=args.patience > 0,
                  parameters=sum(parameter.numel() for parameter in model.module.parameters()),
                  mode=args.mode, epochs=args.epochs, schedule_epochs=schedule_epochs,
                  global_batch=world_size * args.batch_size, local_batch=args.batch_size, world_size=world_size,
                  physical_devices=physical_devices, gradient_accumulation_steps=2, forward_micro_batch=2,
                  resolution=1024, encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=0.01,
                  betas=[0.9, 0.999], seed=42, loss='main_ce_dice_plus_architecture_auxiliary_supervision',
                  scheduler='linear_warmup_poly_0.9', warmup_steps=warmup_steps,
                  amp='bfloat16', attention='torch_sdpa', normalization='fixed_rgb_mean_std',
                  normalization_mean=[0.485, 0.456, 0.406], normalization_std=[0.229, 0.224, 0.225],
                  validation_weights='model',
                  checkpoint_retention='best_and_latest_only',
                  augmentation=AUGMENTATION, decoder_channels=192, decoder_normalization='groupnorm',
                  decoder_dropout=0.0, encoder_drop_path_max=0.1,
                  ignore_index=255,
                  manifest_fingerprint=manifest['fingerprint'],
                  class_names=list(CLASS_NAMES), training_samples=len(training),
                  validation_samples=1599,
                  steps_per_epoch=len(train_loader), torch=torch.__version__, torch_npu=torch_npu.__version__)
    config['source_hashes'] = source_hashes
    config['official_manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    start_epoch, global_step, history = 0, 0, []
    best = dict(epoch=0, mIoU=-1.0)
    extension = None
    if args.resume:
        assert args.resume.resolve() == output / 'latest.pt'
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        extension = validate_resume(checkpoint, config, output, args.extend_epochs, args.redistribute_devices,
                                    allow_wall_time_change=args.unlimited_runtime,
                                    allow_early_stopping_change=args.disable_early_stopping)
        model.module.load_state_dict(checkpoint['model'], strict=True)
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, global_step = checkpoint['epoch'], checkpoint['global_step']
        history, best = checkpoint['history'], checkpoint['best']
        previous_devices = checkpoint['config'].get('physical_devices', list(range(checkpoint['config']['world_size'])))
        saved_rng = checkpoint['rng_states'][previous_devices.index(physical_devices[rank])]
        restore_rng(saved_rng, device)
        restored_rng = rng_state(device)
        assert restored_rng['python'] == saved_rng['python']
        assert restored_rng['numpy'][0] == saved_rng['numpy'][0]
        assert np.array_equal(restored_rng['numpy'][1], saved_rng['numpy'][1])
        assert restored_rng['numpy'][2:] == saved_rng['numpy'][2:]
        assert torch.equal(restored_rng['cpu'], saved_rng['cpu'])
        assert torch.equal(restored_rng['npu'], saved_rng['npu'])
        assert scheduler.last_epoch == global_step
        assert [group['lr'] for group in optimizer.param_groups] == checkpoint['scheduler']['_last_lr']
        atomic_json(attempt / f'restored_state_rank{rank}.json', dict(
            epoch=start_epoch, next_epoch=start_epoch + 1, global_step=global_step, best=best,
            history_entries=len(history), scheduler_last_epoch=scheduler.last_epoch,
            learning_rates=[group['lr'] for group in optimizer.param_groups],
            rng_verified=True, rank=rank, physical_device=physical_devices[rank],
            patience=args.patience, early_stopping_enabled=args.patience > 0))
        del checkpoint
    if rank == 0 and extension is not None:
        archive = output / '.quarantine'
        archive.mkdir(exist_ok=True)
        shutil.copy2(output / 'config.json', archive / f'config_before_resume_{uuid.uuid4().hex}.json')
        record = 'early_stopping_change.json' if args.disable_early_stopping else (
            'wall_time_change.json' if args.unlimited_runtime else 'epoch_extension.json')
        atomic_json(output / record, extension)
        LOGGER.info('续训设置：%s', extension)
    if rank == 0 and (extension is not None or not (output / 'config.json').exists()):
        atomic_json(output / 'config.json', dict(config, fingerprint=fingerprint))
    criterion = SegmentationLoss().to(device)
    started = time.perf_counter()
    step_times = []
    stopping_reason = 'maximum_epochs_completed'
    for epoch_index in range(start_epoch, args.epochs):
        if not args.unlimited_runtime and args.mode == 'selection' and history:
            epoch_cost = max(args.epoch_cost_seconds, max(row['seconds'] for row in history[-3:]))
            deadline_reached = torch.tensor(int(time.time() + epoch_cost + 20 > args.training_deadline)
                                            if rank == 0 else 0, device=device)
            dist.broadcast(deadline_reached, 0)
            if deadline_reached.item():
                epoch = history[-1]['epoch']
                steps = len(train_loader)
                stopping_reason = 'training_wall_time_budget_reached'
                break
        epoch = epoch_index + 1
        sampler.set_epoch(epoch_index)
        model.train()
        dist.barrier()
        epoch_started = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        count = 0
        matrix = torch.zeros((5, 5), device=device, dtype=torch.int64)
        steps = min(args.probe_steps, len(train_loader)) if args.mode == 'probe' else len(train_loader)
        status(output, rank, 'train', epoch, 0, steps, history, best, started, epochs=args.epochs)
        for step, batch in enumerate(train_loader, 1):
            if step > steps:
                break
            if args.mode == 'probe':
                torch.npu.synchronize(device)
            tick = time.perf_counter()
            image = normalize(batch['image'], device)
            target = batch['mask'].to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=torch.float32, device=device)
            assert len(target) % 2 == 0
            for offset in range(2):
                current_target = target[offset::2]
                with model.no_sync() if offset == 0 else nullcontext():
                    with torch.autocast('npu', dtype=torch.bfloat16):
                        output_values = model(image[offset::2])
                        current_loss = criterion(output_values, current_target)
                    (current_loss * 0.5).backward()
                loss += current_loss.detach() * 0.5
                matrix += confusion(output_values['logits'].detach(), current_target)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if step % 20 == 0 or step == 1:
                if not torch.isfinite(loss).item() or not torch.isfinite(norm).item():
                    raise FloatingPointError(f'epoch={epoch} step={step} 的loss或梯度非有限')
            optimizer.step()
            scheduler.step()
            global_step += 1
            loss_sum += loss.detach() * len(target)
            count += len(target)
            if args.mode == 'probe':
                torch.npu.synchronize(device)
                step_times.append(time.perf_counter() - tick)
            if step == 1 or step % 20 == 0 or step == steps:
                status(output, rank, 'train', epoch, step, steps, history, best, started,
                       epochs=args.epochs, loss=loss.item(), global_step=global_step)
        dist.all_reduce(matrix)
        statistics = torch.stack((loss_sum, loss_sum.new_tensor(count)))
        dist.all_reduce(statistics)
        train_report = dict(metrics(matrix.cpu()), loss=(statistics[0] / statistics[1]).item(),
                            samples=int(statistics[1].item()))
        train_seconds = time.perf_counter() - epoch_started
        status(output, rank, 'validation', epoch, 0, len(val_loader), history, best, started)
        validation_report = evaluate(model.module, val_loader, device)
        validation_report['weights'] = 'model'
        assert validation_report['samples'] == 1599
        improved = validation_report['mIoU'] > best['mIoU']
        if improved:
            best = dict(epoch=epoch, mIoU=validation_report['mIoU'], weights='model')
        row = dict(epoch=epoch, global_step=global_step, train=train_report,
                   validation=validation_report, train_seconds=train_seconds,
                   seconds=time.perf_counter() - epoch_started)
        history.append(row)
        states = [None] * world_size
        dist.all_gather_object(states, rng_state(device))
        all_times = [None] * world_size if args.mode == 'probe' else None
        if args.mode == 'probe':
            dist.all_gather_object(all_times, step_times)
        if rank == 0:
            checkpoint = dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(),
                              selected_weights='model',
                              scheduler=scheduler.state_dict(), rng_states=states, epoch=epoch,
                              global_step=global_step, history=history, best=best, config=config,
                              fingerprint=fingerprint)
            publish(checkpoint, output, improved)
            atomic_json(output / 'history.json', history)
            if args.mode == 'probe':
                gate_changes = [(gate.weight.detach().cpu() - initial).abs().max().item()
                                for gate, initial in zip(model.module.encoder.detail_gates, initial_gate_weights)]
                assert all(0 < change < float('inf') for change in gate_changes)
                values = np.max(np.array(all_times), axis=0)[5:]
                atomic_json(output / 'performance.json', dict(median_step_seconds=float(np.median(values)),
                            gate_weight_changes=gate_changes,
                            mean_step_seconds=float(np.mean(values)), p95_step_seconds=float(np.quantile(values, 0.95)),
                            steps=len(values), ranks=world_size,
                            measured_steps=step_times, memory_peak_bytes=torch.npu.max_memory_allocated(device)))
                loaded = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
                assert loaded['fingerprint'] == fingerprint and len(loaded['rng_states']) == world_size
                model.module.load_state_dict(loaded['model'], strict=True)
            LOGGER.info('epoch=%d train_loss=%.6f validation=%s best=%s seconds=%.3f',
                        epoch, train_report['loss'], validation_report, best, row['seconds'])
        dist.barrier()
        if args.mode == 'selection':
            request = [None]
            stop_file = output / 'stop_after_validation.json'
            if rank == 0 and stop_file.is_file():
                request[0] = json.loads(stop_file.read_text())['reason']
                assert isinstance(request[0], str) and request[0].strip()
            dist.broadcast_object_list(request)
            if request[0] is not None:
                stopping_reason = request[0]
                break
        if args.mode == 'selection' and args.patience > 0 and epoch - best['epoch'] >= args.patience:
            stopping_reason = f'validation_mIoU_no_improvement_{args.patience}_epochs'
            LOGGER.info('连续%d轮验证mIoU未提升，停止于第%d轮，最佳第%d轮', args.patience, epoch, best['epoch'])
            break
        if not args.unlimited_runtime and time.perf_counter() - started > args.budget_seconds:
            stopping_reason = 'training_wall_time_budget_reached'
            break
    status(output, rank, 'complete', epoch, steps, steps, history, best, started,
           epochs=args.epochs, stopping_reason=stopping_reason)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
