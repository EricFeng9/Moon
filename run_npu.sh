set -eo pipefail

ROOT=/home/fengjunming/project/Moon/Moon_1115/official_dinov3_subpixel
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$ROOT"
export TMPDIR=/home/fengjunming/project/Moon/Moon_1115/.runtime_tmp/ds
mkdir -p "$TMPDIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HCCL_IF_BASE_PORT=60300
export HCCL_HOST_SOCKET_PORT_RANGE=60400-60463
export HCCL_NPU_SOCKET_PORT_RANGE=60500-60563
ENTRY=${1:?需要指定Python入口}
shift
exec "$ROOT/runtime/bin/python" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
    --master_addr=127.0.0.1 --master_port=30229 -- "$ENTRY" "$@"
