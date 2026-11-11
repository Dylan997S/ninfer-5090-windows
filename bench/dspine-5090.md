# Reproduce the RTX 5090 DSpine benchmark

Use the Windows portable build and a converted Qwen3.8-27B NVFP4 artifact with the DSpine
companion and BF16 lookup tables. The [portable guide](../tools/release/dspine-portable/START-HERE.md)
has the conversion steps. Use an idle RTX 5090 and allow the server to finish loading before timing.

In the extracted portable directory, start the server with the benchmark context and prefill size.

```powershell
$env:NINFER_CUDA_SYNC = 'blocking'
.\ninfer-serve.exe "D:\models\qwen27-dspine-nvfp4-tables.v3.ninfer" `
  --spec dspine --draft-tokens 11 --proposal-head full --kv-dtype fp8 `
  --max-context 2304 --max-concurrency 32 --prefill-chunk 128 `
  --host-cache-capacity 8GiB --no-prefix-reuse --max-pending-requests 32 --pending-timeout 600s `
  --host 127.0.0.1 --port 18092 --model-id qwen27-dspine `
  --chat-template tools/chat_templates/qwen3_8.jinja --gpu-timing
```

In a second shell, from the repository checkout at the release tag, run the Python standard-library
client. The warm-up keeps its result outside the five measured rounds.

```powershell
py -3 bench/dspine-5090.py --workload acceptance --rounds 1
py -3 bench/dspine-5090.py --workload diverse --rounds 5
```

Each round submits 32 simultaneous requests for 2,048 tokens each. The client checks prompt
lengths, exact output lengths, and the engine's width-32 decode counter. The pooled throughput
is total output tokens divided by the sum of measured round wall times, including prefill.

| Measurement | Prompts | Completed | Aggregate output |
|---|---|---:|---:|
| Five-round engineering benchmark | [32 distinct Python prompts](fixtures/dspine-diverse32.json) in each round | 160/160 | 2,368.2 tok/s pooled |
| Single-round portable verification | [Two Python prompts](fixtures/dspine-workload.json) alternated across 32 requests | 32/32 | 2,439.6 tok/s |

Both used Windows 11, one RTX 5090, DSpine K11, FP8 KV, a 2,304-token context, a 128-token
prefill chunk, and prefix reuse disabled. The engineering run ranged from 2,325.9 to 2,424.0
tok/s. The two measurements used different prompt mixes and different builds. Code was timed
and checked for output length. See the [Linux guide](../docs/features/dspine-linux-rtx5090.md)
for the matched Ubuntu 24.04 WSL2 run.
