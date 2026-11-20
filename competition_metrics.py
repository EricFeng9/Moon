import hashlib
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ('regolith', 'rock', 'crater', 'mountain', 'sky')
SCENE_COUNTS = {'Moon_2': 1597, 'Moon_4': 1320, 'Moon_9': 2087}
SCENE_WEIGHTS = {'Moon_2': 1, 'Moon_4': 1, 'Moon_9': 2}
SOURCE_PALETTE = [187, 70, 156, 120, 0, 200, 232, 250, 80, 173, 69, 31, 34, 201, 248]
PROTOCOL = {
    'source': 'https://insight.onescience.cn/competition/moonseg?event_id=c2f557a58ff2e9f4265d56f112a7e01&tab=info&section=overview',
    'page_asset_sha256': '17c20883c64cd7c448593d885c85696053c7d8efbb02f92d82f8bf1132ef29e9',
    'class_names': CLASS_NAMES,
    'source_index_to_competition_id': [0, 2, 1, 3, 4],
    'scene_counts': SCENE_COUNTS,
    'scene_weights': SCENE_WEIGHTS,
    'score_formula': '(mIoU_Moon_2 + mIoU_Moon_4 + 2 * mIoU_Moon_9) / 4',
    'scene_aggregation': 'sum_pixel_confusion_matrices_before_computing_IoU',
    'counted_classes': 'ground_truth_support_greater_than_zero_per_scene',
    'class_weights': 'equal_within_each_scene',
    'image_size': [1024, 1024],
    'prediction_mode': 'L',
    'prediction_ids': [0, 1, 2, 3, 4],
}
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def scene_metrics(matrix):
    matrix = np.asarray(matrix)
    assert matrix.shape == (5, 5) and matrix.dtype == np.int64
    assert matrix.min() >= 0 and matrix.sum() > 0
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = matrix.diagonal()
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    union = true_positive + false_positive + false_negative
    present = support > 0
    iou = np.divide(true_positive, union, out=np.zeros(5, dtype=np.float64), where=union > 0)
    return dict(mIoU=float(iou[present].mean()), IoU=iou.tolist(),
                counted_class_ids=np.flatnonzero(present).tolist(), support=support.tolist(),
                true_positive=true_positive.tolist(), false_positive=false_positive.tolist(),
                false_negative=false_negative.tolist(), confusion_matrix=matrix.tolist())


def competition_score(matrices, counts):
    assert set(matrices) == set(counts) == set(SCENE_COUNTS)
    assert counts == SCENE_COUNTS
    scenes = {}
    for scene in SCENE_COUNTS:
        matrix = np.asarray(matrices[scene], dtype=np.int64)
        assert int(matrix.sum()) == counts[scene] * 1024 * 1024
        scenes[scene] = dict(samples=counts[scene], weight=SCENE_WEIGHTS[scene], **scene_metrics(matrix))
    score = sum(scenes[scene]['mIoU'] * SCENE_WEIGHTS[scene] for scene in scenes) / sum(SCENE_WEIGHTS.values())
    return dict(score=score, scenes=scenes, samples=sum(counts.values()),
                protocol=PROTOCOL, protocol_sha256=PROTOCOL_SHA256)


def labels_for_records(records, label_index):
    assert len(records) == len(label_index) == sum(SCENE_COUNTS.values())
    assert {record['name'] for record in records} == set(label_index)
    result = []
    for record in records:
        assert record['scene'] in SCENE_COUNTS
        path = Path(label_index[record['name']])
        assert path.is_file() and path.name == record['name']
        result.append(dict(record, mask=str(path)))
    assert {scene: sum(record['scene'] == scene for record in result) for scene in SCENE_COUNTS} == SCENE_COUNTS
    return result
