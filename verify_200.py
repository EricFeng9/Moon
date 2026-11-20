import argparse
import math
from pathlib import Path

from common import document, process_matches, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    selection = run / 'selection'
    expected = read(run / 'configuration_changes/2611192316_disable_early_stopping/verified_resume.json')
    config = read(selection / 'config.json')
    status = read(selection / 'status.json')
    driver = read(run / 'status.json')
    assert config['epochs'] == config['schedule_epochs'] == 200
    assert config['patience'] == 0 and config['early_stopping_enabled'] is False
    assert config['physical_devices'] == list(range(8, 16))
    assert driver['phase'] == 'selection' and process_matches(driver['pid'], str(run))
    assert process_matches(status['pid'], 'train.py')
    assert status['phase'] == 'train' and status['epoch'] == expected['next_epoch']
    assert status['step'] >= 20 and status['global_step'] == expected['global_step'] + status['step']
    assert math.isfinite(status['loss'])
    attempts = sorted(selection.glob('attempt_*'))
    attempt = attempts[-1]
    states = [read(attempt / f'restored_state_rank{rank}.json') for rank in range(8)]
    for rank, state in enumerate(states):
        assert state['rank'] == rank and state['physical_device'] == rank + 8
        assert state['epoch'] == expected['epoch'] == state['history_entries']
        assert state['next_epoch'] == expected['next_epoch']
        assert state['global_step'] == expected['global_step'] == state['scheduler_last_epoch']
        assert state['learning_rates'] == expected['learning_rates']
        assert state['best'] == expected['best']
        assert state['rng_verified'] is True
        assert state['patience'] == 0 and state['early_stopping_enabled'] is False
    report = dict(physical_devices=list(range(8, 16)), restored_epoch=expected['epoch'],
                  epoch=status['epoch'], step=status['step'], global_step=status['global_step'],
                  loss=status['loss'], all_eight_ranks_verified=True,
                  epochs=200, patience=0, early_stopping_enabled=False,
                  screen=status['screen'], driver_pid=driver['pid'], rank0_pid=status['pid'],
                  restored_state_directory=str(attempt), best=status['best'],
                  checkpoint_directory=str(selection), estimated_remaining_hours_including_test=13.1)
    write(run / 'continuation_200_health.json', report)
    document(run.name + ' 200轮续训检查通过', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
