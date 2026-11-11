# NVFP4 DSpine on NInfer Windows

This experimental branch adds a native DSpine path for Qwen3.8-27B on the RTX 5090.
It builds on [headpiece747's Windows v1.2.5 port](https://github.com/headpiece747/ninfer-5090-windows/releases/tag/v1.2.5)
of [Neroued/NInfer](https://github.com/Neroued/ninfer), at baseline `5b936c1`.
The draft architecture and five-layer checkpoint are from [Tencent/DSpine](https://github.com/Tencent/DSpine).

The measured preset uses `--spec dspine --draft-tokens 11 --proposal-head full`, FP8 KV,
and optional BF16 lookup tables. Conversion quantizes the draft projections to weight-only
NVFP4. The target tensors are copied and verified by object SHA-256. The full head retrieves
the top 16 candidates and refines them with the DSpine bridge before normal target verification.
The portable launcher defaults to 8,192 context tokens per request and one active request.
The 32-lane build supports explicit concurrency settings up to 32.

## Build and prepare the model

Follow the [Windows build instructions](../../README.md#windows) with CUDA 13.3 and MSVC.
For conversion, use a Python environment with PyTorch, NumPy and safetensors.
Use the v1.2.5 `qwen3_8_27b_nvfp4full_noex.v3.ninfer` artifact as the target.
The benchmark target SHA-256 was
`05f2f48643dcfd1739b5b9624218428c94698491a4981607d01609e5916fcfc9`.
The draft is [tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/tree/ea909c0f5fab8865f9e14600334f55f7ce47308f),
revision `ea909c0f5fab8865f9e14600334f55f7ce47308f`. Obtain `config.json` and `model.safetensors` in the same directory.

```powershell
python -m tools.convert.dspine --base model-base.ninfer --draft draft-hf --out model-dspine.ninfer --projection-format nvfp4
python -m tools.convert.dspine_tables --base model-dspine.ninfer --out model-dspine-tables.ninfer
python -m tools.convert.verify_dspine_tables --base model-dspine.ninfer --tables model-dspine-tables.ninfer --out tables-check.json
```

The final artifact was about 19.8 GiB on disk. Keep sufficient room for both conversion outputs.
Model weights and build products are obtained or built locally.

## Serve the default preset

From the repository root with the default `build` directory:

```powershell
$env:NINFER_CUDA_SYNC = 'blocking'
.\build\apps\ninfer-serve.exe model-dspine-tables.ninfer --spec dspine --draft-tokens 11 --proposal-head full `
  --kv-dtype fp8 --max-context 8192 --context-capacity 8GiB --host-cache-capacity 8GiB `
  --max-concurrency 1 --prefill-chunk 1024 --no-prefix-reuse --host 127.0.0.1 --port 18092 `
  --model-id qwen27-dspine --chat-template tools/chat_templates/qwen3_8.jinja
```

See the [portable guide](../../tools/release/dspine-portable/START-HERE.md) for the optional
32-lane configuration. Context storage is shared across active requests.

## Historical RTX 5090 measurements

Same target tensors, Windows, two prompts per workload, greedy text generation,
seed 42, thinking disabled, presence penalty 1.5, frequency penalty 0, and no prefix reuse.
The [workload fixture](../../bench/fixtures/dspine-workload.json) contains the exact prompts.
At concurrency eight the two prompts alternate across the eight simultaneous requests.
Single-stream speed is decode tokens per server decode second. Eight-stream throughput is total
output tokens per client batch second. These results are from the prior experiment with the
same preset. The focused branch verification is recorded separately.

| Workload | Output tokens | Streams | DFlash2 K9 | NVFP4 DSpine K11 |
|---|---:|---:|---:|---:|
| Python code | 256 | 1 | 324.5 | 377.0 |
| Python code | 256 | 8 | 957.6 | 961.2 |
| Python code | 2048 | 1 | 422.8 | 514.9 |
| Python code | 2048 | 8 | 1570.4 | 1674.8 |
| English prose | 2048 | 1 | 183.0 | 160.1 |
| English prose | 2048 | 8 | 810.5 | 684.4 |
| Chinese prose | 2048 | 1 | 131.5 | 152.0 |
| Chinese prose | 2048 | 8 | 552.7 | 624.0 |
| ~1.7K input analysis | 2048 | 1 | 225.8 | 230.5 |
| ~1.7K input analysis | 2048 | 8 | 889.0 | 947.4 |

All rates are tokens/s. The code rows use the later two-run single-stream and five-run
eight-stream series. Eight-stream code ranges overlapped. For the ~1.7K-token prompts,
single-stream prefill was 13,003 vs 12,960 tokens/s. Recorded weight residency was
16.6 vs 17.5 GiB. The tests measured generation speed, and did not grade or execute the outputs.

## Branch verification

A fresh Windows x64 build with MSVC 19.44 and CUDA 13.3 passed eight focused CTest cases
covering the CLI, DSpine and its tables, NVFP4 linear add and pair, Q8 linear pair,
KV append, and sliding-window attention. Nine converter regression tests and sampled
FP64-reference checks of both stored tables passed. A real-model K11 smoke run generated
the requested 256 tokens in each of nine requests (one English request and eight concurrent
code requests). All nine recorded accepted DSpine draft tokens. The full release suite
was outside this branch verification.
