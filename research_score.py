from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import PROJECT, atomic_json, digest, document, read
from competition_metrics import PROTOCOL_SHA256, SCENE_COUNTS, competition_score, labels_for_records, scene_metrics
from verify_test_predictions import matrix_for_record


def score_pixels(predictions, index):
    records = labels_for_records(read(PROJECT/'data/manifest.json')['test'], read(index))
    expected = {predictions/'test'/r['scene']/'submission'/r['name'] for r in records}
    assert expected == set((predictions/'test').rglob('*.png'))
    matrices = {scene: np.zeros((5, 5), dtype=np.int64) for scene in SCENE_COUNTS}
    counts = dict.fromkeys(SCENE_COUNTS, 0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for scene, matrix in pool.map(matrix_for_record, ((r, predictions) for r in records)):
            matrices[scene] += matrix
            counts[scene] += 1
    result = competition_score(matrices, counts)
    result['pooled'] = scene_metrics(sum(matrices.values()))
    return result


def score(run):
    report = read(run/'prediction_report.json')
    checkpoint = run/'selection/best.pt'
    assert report['images'] == sum(SCENE_COUNTS.values())
    assert report['checkpoint_sha256'] == digest(checkpoint)
    assert report['best'] == read(run/'selection/status.json')['best']
    index = PROJECT/'data/sealed_test_labels.json'
    output = run/'public_research_score.json'
    if output.exists():
        previous = read(output)
        assert previous['checkpoint_sha256'] == report['checkpoint_sha256']
        assert previous['protocol_sha256'] == PROTOCOL_SHA256
        assert previous['label_index_sha256'] == digest(index)
        return previous
    predictions = Path(report['predictions'])
    result = score_pixels(predictions, index)
    result.update(best=report['best'],
                  checkpoint=str(checkpoint), checkpoint_sha256=report['checkpoint_sha256'],
                  predictions=str(predictions), label_index_sha256=digest(index),
                  official_submission_eligible=False, purpose='independent_research_architecture_selection')
    document(run.name+' 公开测试集独立研究评分', result)
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('run', type=Path)
    print(score(parser.parse_args().run)['score'])
