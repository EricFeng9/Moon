import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image

from competition_metrics import SCENE_COUNTS, SOURCE_PALETTE, competition_score, labels_for_records
from data import SOURCE_TO_COMPETITION


ROOT = Path(__file__).resolve().parent


def matrix_for_record(item):
    record, prediction_root = item
    prediction_path = prediction_root / 'test' / record['scene'] / 'submission' / record['name']
    with Image.open(prediction_path) as image:
        assert image.format == 'PNG' and image.mode == 'L' and image.size == (1024, 1024)
        prediction = np.asarray(image)
        assert prediction.dtype == np.uint8 and prediction.max() < 5
    with Image.open(record['mask']) as image:
        assert image.mode == 'P' and image.size == (1024, 1024)
        assert image.getpalette()[:15] == SOURCE_PALETTE
        target = np.asarray(image)
        assert target.max() < 5
        target = SOURCE_TO_COMPETITION[target]
    encoded = target.astype(np.int64).reshape(-1) * 5 + prediction.reshape(-1)
    return record['scene'], np.bincount(encoded, minlength=25).reshape(5, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--research-only', action='store_true', required=True)
    args = parser.parse_args()
    assert args.research_only
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    labels = json.loads((ROOT / 'data/sealed_test_labels.json').read_text())
    records = labels_for_records(manifest['test'], labels)
    expected = {f"test/{record['scene']}/submission/{record['name']}" for record in records}
    actual = {str(path.relative_to(args.predictions)) for path in (args.predictions / 'test').rglob('*.png')}
    assert actual == expected
    matrices = {scene: np.zeros((5, 5), dtype=np.int64) for scene in SCENE_COUNTS}
    counts = {scene: 0 for scene in SCENE_COUNTS}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for scene, matrix in pool.map(matrix_for_record, ((record, args.predictions) for record in records)):
            matrices[scene] += matrix
            counts[scene] += 1
    result = competition_score(matrices, counts)
    reference = json.loads((args.predictions / 'result.json').read_text())
    assert result['protocol_sha256'] == reference['protocol_sha256']
    for scene in SCENE_COUNTS:
        assert result['scenes'][scene]['confusion_matrix'] == reference['scenes'][scene]['confusion_matrix']
        assert result['scenes'][scene]['counted_class_ids'] == reference['scenes'][scene]['counted_class_ids']
        assert abs(result['scenes'][scene]['mIoU'] - reference['scenes'][scene]['mIoU']) < 1e-12
    assert abs(result['score'] - reference['score']) < 1e-12
    result.update(independent_cpu_recalculation=True, all_pixels_equal_to_npu=True)
    with (args.predictions / 'independent_verification.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(samples=result['samples'], score=result['score'], all_pixels_equal=True)))


if __name__ == '__main__':
    main()
