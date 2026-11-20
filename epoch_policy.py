import hashlib
import json
from pathlib import Path

from common import digest


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def verify_sources(config, folder):
    expected = config['source_hashes']
    matches = [source for source in Path(folder).glob('attempt_*/source')
               if all((source / name).is_file() and digest(source / name) == identity
                      for name, identity in expected.items())]
    assert matches, (folder, expected)
    return str(sorted(matches)[-1])


def validate_resume(checkpoint, config, folder, allow_extension, allow_devices=False, *, allow_wall_time_change=False,
                    allow_early_stopping_change=False):
    previous = checkpoint['config']
    assert checkpoint['fingerprint'] == fingerprint(previous)
    if checkpoint['fingerprint'] == fingerprint(config):
        return None
    if allow_early_stopping_change:
        assert config['mode'] == previous['mode'] == 'selection'
        assert config['epochs'] == previous['epochs'] == config['schedule_epochs'] == 200
        assert previous['patience'] == 20 and config['patience'] == 0
        assert config['early_stopping_enabled'] is False
        mutable = {'patience', 'early_stopping_enabled', 'source_hashes'}
        assert {k:v for k,v in previous.items() if k not in mutable} == {
            k:v for k,v in config.items() if k not in mutable}
        old_sources, new_sources = previous['source_hashes'], config['source_hashes']
        assert set(old_sources) == set(new_sources)
        changed = {name for name in old_sources if old_sources[name] != new_sources[name]}
        assert changed <= {'train.py', 'epoch_policy.py', 'run.py'}, changed
        assert len(checkpoint['rng_states']) == config['world_size'] == 8
        assert checkpoint['global_step'] == checkpoint['epoch'] * config['steps_per_epoch']
        return dict(policy='complete_200_epochs_without_early_stopping', epoch=checkpoint['epoch'],
                    global_step=checkpoint['global_step'], patience=0, early_stopping_enabled=False,
                    verified_source_snapshot=verify_sources(previous, folder),
                    previous_fingerprint=checkpoint['fingerprint'], fingerprint=fingerprint(config),
                    unchanged_other_training_settings=True,
                    restored_state=['model', 'optimizer', 'scheduler', 'rng_states', 'global_step', 'history', 'best'])
    if allow_wall_time_change:
        assert config['mode'] == 'selection'
        old_sources, new_sources = previous['source_hashes'], config['source_hashes']
        assert {k:v for k,v in previous.items() if k != 'source_hashes'} == {
            k:v for k,v in config.items() if k != 'source_hashes'}
        assert set(old_sources) == set(new_sources)
        changed = {name for name in old_sources if old_sources[name] != new_sources[name]}
        assert changed <= {'run.py', 'train.py', 'epoch_policy.py'}, changed
        return dict(policy='authorized_wall_time_limit_removal', epoch=checkpoint['epoch'],
                    verified_source_snapshot=verify_sources(previous, folder),
                    previous_fingerprint=checkpoint['fingerprint'], fingerprint=fingerprint(config),
                    restored_state=['model', 'optimizer', 'scheduler', 'rng_states', 'global_step', 'history', 'best'])
    if allow_devices:
        assert previous['world_size'] == 16 and previous['local_batch'] == 2
        assert config['world_size'] == 8 and config['local_batch'] == 4
        assert previous['global_batch'] == config['global_batch'] == 32
        assert config['physical_devices'] == list(range(8, 16))
        assert config['gradient_accumulation_steps'] == config['forward_micro_batch'] == 2
        mutable = {'world_size', 'local_batch', 'source_hashes', 'physical_devices',
                   'gradient_accumulation_steps', 'forward_micro_batch'}
        assert {key: value for key, value in previous.items() if key not in mutable} == {
            key: value for key, value in config.items() if key not in mutable}
        assert set(previous['source_hashes']) == set(config['source_hashes'])
        changed = {name for name in previous['source_hashes']
                   if previous['source_hashes'][name] != config['source_hashes'][name]}
        assert changed <= {'data.py', 'train.py', 'evaluation.py', 'run_npu.sh', 'epoch_policy.py'}, changed
        assert len(checkpoint['rng_states']) == 16
        return dict(policy='authorized_device_redistribution', epoch=checkpoint['epoch'],
                    previous_world_size=16, world_size=8, physical_devices=config['physical_devices'],
                    global_batch=32, local_batch=4, gradient_accumulation_steps=2, forward_micro_batch=2,
                    rng_source_ranks=list(range(8, 16)),
                    verified_source_snapshot=verify_sources(previous, folder),
                    previous_fingerprint=checkpoint['fingerprint'], fingerprint=fingerprint(config),
                    unchanged_other_training_settings=True)
    assert allow_extension and config['mode'] == 'selection'
    assert config['epochs'] == config['schedule_epochs'] == 200
    assert config['patience'] == previous['patience'] == 10
    assert previous['epochs'] <= 200 and previous['schedule_epochs'] <= 200
    mutable = {'epochs', 'schedule_epochs', 'source_hashes'}
    assert {key: value for key, value in previous.items() if key not in mutable} == {
        key: value for key, value in config.items() if key not in mutable}
    old_sources, new_sources = previous['source_hashes'], config['source_hashes']
    assert set(new_sources) == set(old_sources) | {'epoch_policy.py'}
    changed = {name for name in old_sources if old_sources[name] != new_sources[name]}
    assert changed <= {'train.py', 'evaluation.py'}, changed
    source = verify_sources(previous, folder)
    return dict(policy='authorized_epoch_extension', epoch=checkpoint['epoch'],
                previous_epochs=previous['epochs'], maximum_epochs=200,
                previous_schedule_epochs=previous['schedule_epochs'], schedule_epochs=200,
                patience=10, previous_fingerprint=checkpoint['fingerprint'],
                fingerprint=fingerprint(config), verified_source_snapshot=source,
                unchanged_training_configuration=True,
                restored_state=['model', 'ema_model', 'optimizer', 'scheduler', 'rng_states',
                                'global_step', 'history', 'best'])
