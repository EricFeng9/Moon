import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


root = Path(__file__).resolve().parents[1]
dependency = json.loads((root / 'dependencies.json').read_text())
folder = root / 'weights'
folder.mkdir(exist_ok=True)
target = folder / 'dinov3_vits16_pretrain_lvd1689m-08c60483.bin'
if not target.exists():
    pending = folder / 'dinov3_vits16_pretrain_lvd1689m-08c60483.download'
    with pending.open('xb') as output:
        with urllib.request.urlopen(dependency['download'], timeout=120) as response:
            shutil.copyfileobj(response, output)
    with pending.open('rb') as stream:
        identity = hashlib.file_digest(stream, 'sha256').hexdigest()
    if identity != dependency['sha256']:
        raise ValueError('下载文件的 SHA256 与预训练权重记录不一致')
    pending.rename(target)
with target.open('rb') as stream:
    identity = hashlib.file_digest(stream, 'sha256').hexdigest()
if identity != dependency['sha256']:
    raise ValueError('预训练权重的 SHA256 与记录不一致')
print(f'权重校验通过：{target}')
