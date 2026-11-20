import argparse
import os
from pathlib import Path
import shutil
import signal

from common import ROOT, atomic_json, digest, document, process_matches, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('request', 'stop-driver', 'configure'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--change-id', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.name == '261119_moon_dinov3_subpixel'
    selection = run / 'selection'
    change = run / 'configuration_changes' / args.change_id
    if args.action == 'request':
        driver = read(run / 'status.json')
        assert driver['phase'] == 'selection'
        assert process_matches(driver['pid'], str(run))
        assert not (selection / 'stop_after_validation.json').exists()
        change.mkdir(parents=True)
        for name in ('runtime_policy.json', 'protocol.json', 'status.json'):
            shutil.copy2(run / name, change / name)
        previous_source = change / 'source'
        previous_source.mkdir()
        for name in ('train.py', 'run.py', 'epoch_policy.py'):
            shutil.copy2(ROOT / name, previous_source / name)
        write(change / 'request.json', dict(driver_pid=driver['pid'],
              goal_epochs=200, patience=0, early_stopping_enabled=False,
              current_status=read(selection / 'status.json')))
        os.kill(driver['pid'], signal.SIGSTOP)
        atomic_json(selection / 'stop_after_validation.json',
                    dict(reason='checkpoint_boundary_for_200_epoch_continuation'))
        print(change, flush=True)
        return

    if args.action == 'stop-driver':
        pid = read(change / 'request.json')['driver_pid']
        assert process_matches(pid, str(run))
        write(change / 'driver_termination.json', dict(pid=pid, reason='continue_current_run_to_200_epochs'))
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        print('driver termination requested', flush=True)
        return

    import torch
    from epoch_policy import validate_resume

    status = read(selection / 'status.json')
    assert status['phase'] == 'complete'
    assert status['stopping_reason'] == 'checkpoint_boundary_for_200_epoch_continuation'
    assert not process_matches(status['pid'], 'train.py')
    checkpoint_path = selection / 'latest.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == status['completed_epoch'] == len(checkpoint['history'])
    config = dict(checkpoint['config'], patience=0, early_stopping_enabled=False)
    config['source_hashes'] = {name: digest(ROOT / name) for name in checkpoint['config']['source_hashes']}
    verified = validate_resume(checkpoint, config, selection, False, allow_early_stopping_change=True)
    write(change / 'verified_resume.json', dict(verified, latest_sha256=digest(checkpoint_path),
          best_sha256=digest(selection / 'best.pt'), best=checkpoint['best'],
          checkpoint=str(checkpoint_path), next_epoch=checkpoint['epoch'] + 1,
          learning_rates=checkpoint['scheduler']['_last_lr']))
    request = read(change / 'request.json')
    pid = request['driver_pid']
    assert read(change / 'driver_termination.json')['pid'] == pid
    assert not process_matches(pid, str(run))
    (selection / 'stop_after_validation.json').rename(change / 'stop_after_validation.json')
    policy = dict(max_seconds=None, epochs=200, patience=0, early_stopping_enabled=False)
    atomic_json(run / 'runtime_policy.json', policy)
    protocol = read(run / 'protocol.json')
    protocol.update(policy)
    protocol['continuation'] = dict(checkpoint=str(checkpoint_path), epoch=checkpoint['epoch'],
                                    state_verification=str(change / 'verified_resume.json'))
    atomic_json(run / 'protocol.json', protocol)
    atomic_json(run / 'status.json', dict(phase='ready_to_resume', output=str(selection),
                completed_epoch=checkpoint['epoch'], goal_epochs=200, early_stopping_enabled=False))
    document(run.name + ' 完整训练200轮设置', dict(policy, physical_devices=list(range(8, 16)),
             completed_epoch=checkpoint['epoch'], next_epoch=checkpoint['epoch'] + 1,
             restored_state=verified['restored_state'], verification=str(change / 'verified_resume.json'),
             checkpoint_directory=str(selection), validation='每轮完整验证Moon5并选择best.pt',
             test='完成200轮后使用best.pt进行完整六视图测试'))
    print(read(change / 'verified_resume.json'), flush=True)


if __name__ == '__main__':
    main()
