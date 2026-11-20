set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
PYTHON=${MOON_PYTHON:?请设置 MOON_PYTHON 为 NPU 环境的 Python 路径}
RUN=${1:?需要指定新的实验目录}
MANIFEST=${2:?需要指定训练 manifest.json}
MODE=${3:-}
test "$(basename "$MANIFEST")" = manifest.json
MANIFEST=$(cd "$(dirname "$MANIFEST")" && pwd -P)/manifest.json
test -f "$MANIFEST"
if [ "$MODE" = --resume ]; then
    test -f "$RUN/selection/latest.pt"
elif [ -n "$MODE" ]; then
    exit 2
else
    test ! -e "$RUN"
    mkdir -p "$RUN"
    ln -s "$(dirname "$MANIFEST")" "$RUN/official_train"
fi
RUN=$(cd "$RUN" && pwd -P)
test "$(cd "$RUN/official_train" && pwd -P)/manifest.json" = "$MANIFEST"
cd "$ROOT"
export TMPDIR="$ROOT/.runtime_tmp/train"
mkdir -p "$TMPDIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ARGS=(--output-dir "$RUN/selection" --manifest "$MANIFEST" --epochs 200 --schedule-epochs 200
      --mode selection --variant dinov3_subpixel_e018 --batch-size 4
      --patience 0 --disable-early-stopping --unlimited-runtime)
if [ "$MODE" = --resume ]; then
    ARGS+=(--resume "$RUN/selection/latest.pt")
fi
exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
    --master_addr=127.0.0.1 --master_port="${MOON_MASTER_PORT:-29578}" \
    train.py "${ARGS[@]}"
