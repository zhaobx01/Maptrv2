#!/usr/bin/env bash


MASTER_ADDR=${1:-localhost}
MASTER_PORT=6000
NNODES=${2:-1}
NODE_RANK=${3:-0}
GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CONFIG=$4

# add numa affinity config
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun $DISTRIBUTED_ARGS --no-python \
    bash -c '
    numa_map=( $(hy-smi --showtopo | grep "Numa Node" | awk "{print \$6}") );
    LOCAL_RANK=${LOCAL_RANK:-0}
    NUMA_ID=${numa_map[$LOCAL_RANK]}
    numactl --cpunodebind=${NUMA_ID} --membind=${NUMA_ID} python $(dirname "$0")/tools/train.py "$@"
    ' _ $CONFIG --launcher pytorch ${@:5} --deterministic
