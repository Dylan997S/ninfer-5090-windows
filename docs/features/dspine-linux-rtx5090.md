# NInfer DSpine for RTX 5090 on Linux

The Linux target uses the same Qwen3.8-27B NVFP4 artifact, Tencent DSpine draft, and
BF16 lookup tables as the Windows build. See the [conversion guide](dspine-nvfp4-windows.md#build-and-prepare-the-model)
for the pinned model and conversion commands.

## Build

Use Linux x86-64, an NVIDIA driver compatible with CUDA 13.4, CMake 3.28 or later,
Ninja, and a CUDA-compatible C++20 compiler. On Ubuntu 24.04 the system development
dependencies are `pkg-config`, `libhwloc-dev`, `liburing-dev`, `libjpeg-dev`,
`libavformat-dev`, `libavcodec-dev`, `libavutil-dev`, and `libswscale-dev`.
`libcurl4-openssl-dev` enables remote HTTP media acquisition.

```bash
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
cmake -S . -B build/rtx5090-linux -G Ninja \
  -DNINFER_TARGET=rtx5090 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DBUILD_TESTING=OFF
cmake --build build/rtx5090-linux --parallel
```

## Run

From a Linux package, pass the converted model to `./start_dspine_nvfp4.sh`.
The launcher defaults to one active request, 8,192 context tokens, FP8 KV, and a
1,024-token prefill chunk. Set `CUDA_HOME` if CUDA is installed outside
`/usr/local/cuda`. The loopback endpoint is `http://127.0.0.1:18092/v1`.
On Ubuntu 24.04, the package expects the system libraries `libhwloc15`, `liburing2`,
`libjpeg-turbo8`, `libavformat60`, `libavcodec60`, `libavutil58`, `libswscale7`,
and `libcurl4t64`, plus the compatible CUDA runtime and driver.

Enable 32 active lanes with automatic Device context capacity and a 128-token prefill chunk.

```bash
./start_dspine_nvfp4.sh /path/to/qwen27-dspine-nvfp4-tables.v3.ninfer \
  --max-concurrency 32 --context-capacity auto --prefill-chunk 128 \
  --max-pending-requests 32 --pending-timeout 600s --gpu-timing
```

For the throughput benchmark, add `--max-context 2304` and run the same
[32-request fixture and client](../../bench/dspine-5090.md) as on Windows.
The runtime shares context memory across active requests.

## Ubuntu 24.04 WSL2 verification

One RTX 5090, CUDA 13.4, GCC 13, DSpine K11, FP8 KV, and disabled prefix reuse.
The Windows and WSL2 workloads used the same Qwen3.8-27B NVFP4 model objects,
chat template, prompts, and output lengths.

| Workload | Windows | Ubuntu 24.04 WSL2 |
|---|---:|---:|
| Single-request English decode, 2,048 output tokens, 32-lane limit | 155.5 tok/s | 154.0 tok/s |
| Single-request Python decode, 2,048 output tokens, 32-lane limit | 511.0 tok/s | 510.3 tok/s |
| 1,716-token input prefill, default one-lane profile | 12,790 tok/s | 12,231 tok/s |
| 1,716-token input prefill, 32-lane profile and one active request | 6,487 tok/s | 6,241 tok/s |
| 32 simultaneous requests, two Python prompts, one round | 2,439.6 tok/s | 2,345.6 tok/s |
| 32 distinct Python prompts per round, five rounds | 2,368.2 tok/s | 2,349.1 tok/s |

The wide benchmark used a 2,304-token context and 2,048 output tokens per request.
WSL2 completed 32/32 in the single round and 160/160 in the five-round run, with
274 and 1,181 actual width-32 decode steps, respectively. The 8,192-token mixed-request
regression completed 32/32 with streaming, nonstreaming, long-input retrieval, and tool calls.
With the 32-lane limit enabled and one active request, the 262,144-token configuration
processed a 262,064-token input and retrieved all three planted values, with 83.9 seconds
to the first token. The Linux direct-read queue regression also passed.
Bare-metal Linux is unverified.
