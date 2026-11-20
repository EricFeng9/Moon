import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
expected = json.loads((root / 'records/source_checksums.json').read_text())
for name, identity in expected.items():
    with (root / name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != identity:
        raise ValueError(f'{name} 的 SHA256 与导出记录不一致')
print(f'已校验 {len(expected)} 个原始源码与配置文件')
