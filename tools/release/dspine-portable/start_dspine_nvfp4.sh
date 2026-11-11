#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model="$root/models/qwen27-dspine-nvfp4-tables.v3.ninfer"
if (( $# > 0 )); then
    model=$1
    shift
fi
if [[ ! -f "$model" ]]; then
    printf 'Model not found: %s\n' "$model" >&2
    exit 2
fi

cuda=${CUDA_HOME:-/usr/local/cuda}
runtime="$root/lib:$cuda/lib64:$cuda/lib"
if [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
    runtime="$runtime:/usr/lib/wsl/lib"
fi
export LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NINFER_CUDA_SYNC=${NINFER_CUDA_SYNC:-blocking}

exec "$root/apps/ninfer-serve" "$model" \
    --spec dspine --draft-tokens 11 --proposal-head full --kv-dtype fp8 \
    --max-context 8192 --context-capacity 8GiB --host-cache-capacity 8GiB \
    --max-concurrency 1 --prefill-chunk 1024 --no-prefix-reuse \
    --host 127.0.0.1 --port 18092 --model-id qwen27-dspine \
    --chat-template "$root/tools/chat_templates/qwen3_8.jinja" "$@"
