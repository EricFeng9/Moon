import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from common import PROJECT, ROOT, atomic_json, document, read
from research_score import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run.resolve()
    manifest = run / 'official_train/manifest.json'
    assert len(read(manifest)['train']) == 6403
    lock = (PROJECT/'architecture_search.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run/'prediction_report.json').is_file() and (run/'status.json').is_file() and read(run/'status.json')['phase']=='complete':
        return
    startfile = run/'started.json'
    if not startfile.exists():
        atomic_json(startfile,dict(start=time.time()))
    started = read(startfile)['start']
    policy = run/'runtime_policy.json'
    assert read(policy) == dict(max_seconds=None, epochs=200, patience=0, early_stopping_enabled=False)
    maximum_seconds = read(policy)['max_seconds'] if policy.is_file() else 43200
    deadline = started+maximum_seconds if maximum_seconds is not None else None
    training_deadline = deadline-3600 if deadline is not None else None
    def status(phase, **kw):
        atomic_json(run/'status.json',dict(phase=phase,updated=time.time(),pid=os.getpid(),**kw))
    def launch(command, until):
        with (run/'driver.log').open('a') as log:
            process = subprocess.Popen(['bash',str(ROOT/'run_npu.sh'),*command],cwd=ROOT,start_new_session=True,
                                       stdout=log,stderr=subprocess.STDOUT)
            try:
                timeout = max(1,until-time.time()) if until is not None else None
                code = process.wait(timeout=timeout)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    process.wait(timeout=45)
            if code:
                raise subprocess.CalledProcessError(code,command)
    record=run/'protocol.json'
    if not record.exists():
        spec=dict(architecture='DINOv3 gated encoder with RGB-guided subpixel residual logits',
                  parent=read(run/'parent.json'), cpu_verification=read(run/'parent_verification.json'),
                  pretrained=read(ROOT/'dependencies.json'),
                  source=['https://arxiv.org/abs/2510.12764','https://arxiv.org/abs/1609.05158'],
                  max_time_ratio=1.10, baseline_parameters=29222843,
                  seed=42,training_package=read(manifest)['source'],train_scenes=[1,3,6,7,8],validation_scene=5,
                  training_samples=6403, validation_samples=1599, sampling='uniform distributed shuffle',
                  inputs=1024,devices=list(range(8,16)),batch=32,optimizer='AdamW',lr=[0.0003,0.0006],
                  loss='complete E018 criterion',initialization='frozen DINOv3, random trainable layers',
                  selection='Moon5 mIoU ordinary weights',patience=0,early_stopping_enabled=False,
                  epochs=200,max_seconds=maximum_seconds,test='official unlabelled RGB / six views',run=str(run),
                  checkpoints=[str(run/'selection'/name) for name in ('best.pt','latest.pt')])
        document(run.name+' 门控模型的原尺寸像素细节实验',spec)
        atomic_json(record,spec)
    phase='benchmark'
    try:
        status(phase)
        if not (run/'benchmark.json').exists():
            launch(['benchmark.py',str(run)],training_deadline)
        bench=read(run/'benchmark.json')
        if 'selected' not in bench:
            status('speed_review',measurements=str(run/'benchmark.json'))
            document(run.name+' 速度测量', bench[read(run/'parent.json')['variant']])
            return
        variant=bench['selected']
        if not (run/'choice.json').exists():
            choice=dict(selected=variant,within_speed_target=bench['within_speed_target'],**bench[variant])
            document(run.name+' 训练与预测速度及选用结构',choice)
            atomic_json(run/'choice.json',choice)
        epoch_seconds=0
        for phase,epochs in [('probe',1),('selection',200)]:
            folder = run/phase
            status(phase,output=str(folder))
            finished = read(folder/'status.json') if (folder/'status.json').exists() else None
            if finished is None or finished['phase'] != 'complete' or (
                    phase == 'selection' and finished['completed_epoch'] < epochs):
                command=['train.py','--output-dir',str(folder),'--epochs',str(epochs),'--schedule-epochs','200',
                         '--mode',phase,'--variant',variant,'--manifest',str(manifest)]
                if phase == 'selection':
                    command.extend(['--patience','0','--disable-early-stopping'])
                else:
                    command.extend(['--patience','20'])
                if training_deadline is None:
                    command.append('--unlimited-runtime')
                else:
                    command.extend(['--training-deadline',str(training_deadline),
                                    '--budget-seconds',str(int(training_deadline-time.time())),
                                    '--epoch-cost-seconds',str(epoch_seconds)])
                if (folder/'latest.pt').exists():
                    command.extend(['--resume',str(folder/'latest.pt')])
                elif folder.exists():
                    folder.rename(run/f'{phase}_interrupted_{time.time_ns()}')
                launch(command,training_deadline+100 if training_deadline is not None else None)
            if phase=='probe':
                row=read(folder/'history.json')[0]
                assert row['global_step']==100 and row['validation']['samples']==1599
                epoch_seconds=read(folder/'performance.json')['p95_step_seconds']*201+row['validation']['seconds']+20
        finished=read(run/'selection/status.json')
        assert finished['completed_epoch'] == 200 and finished['stopping_reason'] == 'maximum_epochs_completed'
        assert read(run/'probe/config.json')['random_initialization_digests']==read(run/'selection/config.json')['random_initialization_digests']
        phase='official_test_inference'
        status(phase)
        if not (run/'prediction_report.json').exists():
            if (run/'predictions').exists():
                (run/'predictions').rename(run/f'predictions_interrupted_{time.time_ns()}')
            launch(['predict.py',str(run)],deadline)
        report=read(run/'prediction_report.json')
        report.update(stopping_reason=read(run/'selection/status.json')['stopping_reason'],seconds=time.time()-started,
                      variant=variant,speed=bench[variant],baseline_best_Moon5=0.8816786188586656,
                      delta_Moon5_points=100*(report['best']['mIoU']-0.8816786188586656))
        phase='public_research_scoring'
        status(phase)
        result=score(run)
        parent=read(run/'parent.json')
        promote=result['score'] > parent['score']
        decision=dict(candidate=variant, score=result['score'], parent=parent['identifier'],
                      parent_score=parent['score'], promoted=promote,
                      incumbent=variant if promote else parent['identifier'],
                      checkpoint=str(run/'selection/best.pt') if promote else parent['checkpoint'],
                      score_file=str(run/'public_research_score.json') if promote else parent['score_file'],
                      official_submission_eligible=False)
        atomic_json(run/'architecture_decision.json',decision)
        report.update(research_decision=decision,seconds=time.time()-started)
        document(run.name+' 验证与独立研究测试完成',report)
        status('complete',report=report)
    except Exception as error:
        status('failed',stage=phase,error=str(error))
        raise


if __name__=='__main__':
    main()
