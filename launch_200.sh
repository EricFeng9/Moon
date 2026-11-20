set -eo pipefail

ROOT=/home/fengjunming/project/Moon/Moon_1115/official_dinov3_subpixel
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$ROOT"
export TMPDIR=/home/fengjunming/project/Moon/Moon_1115/.runtime_tmp/ds
mkdir -p "$TMPDIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec "$ROOT/runtime/bin/python" -u run.py "${1:?需要指定实验结果目录}"
