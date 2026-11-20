from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.pending')
    write(pending, value)
    os.replace(pending, path)


def document(title, value):
    text = f'\n## {title}\n\n```json\n' + json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n```\n'
    with (PROJECT / 'moon实验方案.md').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(text)
        stream.flush()


def now():
    return datetime.now(timezone.utc).isoformat()


def process_matches(pid, expected):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and expected.encode() in path.read_bytes()
