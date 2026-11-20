set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
PYTHON=${MOON_PYTHON:?请设置 MOON_PYTHON 为 NPU 环境的 Python 路径}
RUN=${1:?需要指定实验目录}
RUN=$(cd "$RUN" && pwd -P)
test -f "$RUN/selection/best.pt"
test -f "$RUN/official_train/manifest.json"
test ! -e "$RUN/prediction_report.json"
test ! -e "$RUN/predictions"
cd "$ROOT"
export TMPDIR="$ROOT/.runtime_tmp/predict"
mkdir -p "$TMPDIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
    --master_addr=127.0.0.1 --master_port="${MOON_MASTER_PORT:-29579}" \
    predict.py "$RUN"
