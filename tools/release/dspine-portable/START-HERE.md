# NInfer DSpine for RTX 5090

Windows 11 x64 portable build for Qwen3.8-27B NVFP4 on one RTX 5090. It is based on
[headpiece747's NInfer Windows port](https://github.com/headpiece747/ninfer-5090-windows)
of [Neroued/NInfer](https://github.com/Neroued/ninfer). The DSpine architecture and draft
checkpoint are from [Tencent/DSpine](https://github.com/Tencent/DSpine).
[Dylan](https://github.com/Dylan997S) integrated DSpine into the Windows engine,
enabled 32 active execution lanes, and reduced DSpine's state and draft KV memory.

## Start with a converted model / 启动

Extract the entire ZIP and keep its DLLs beside the executables. In PowerShell, from
the extracted folder, pass a converted model:

```powershell
.\start_dspine_nvfp4.bat "D:\models\qwen27-dspine-nvfp4-tables.v3.ninfer"
```

Without a model argument, the launcher reads
`models\qwen27-dspine-nvfp4-tables.v3.ninfer` relative to the extracted folder.
The OpenAI-compatible API is `http://127.0.0.1:18092/v1` with model ID `qwen27-dspine`.
Run `curl.exe http://127.0.0.1:18092/health` to check readiness. Press Ctrl+C to stop.
Serving needs a compatible NVIDIA driver. The bundled DLLs allow the server to run without a
CUDA Toolkit, Visual Studio, or Python installation.

The launcher defaults to one active request and an 8,192-token context per request,
including input and output. It uses text-only DSpine K11 with the full proposal head
and FP8 KV.

To opt into the tested 32-lane configuration, start the same executable with a smaller
prefill chunk and automatic Device context capacity. Close any server already using port 18092 first.

```powershell
$env:NINFER_CUDA_SYNC = 'blocking'
.\ninfer-serve.exe "D:\models\qwen27-dspine-nvfp4-tables.v3.ninfer" `
  --spec dspine --draft-tokens 11 --proposal-head full --kv-dtype fp8 `
  --max-context 8192 --max-concurrency 32 --prefill-chunk 128 `
  --host-cache-capacity 8GiB --no-prefix-reuse --max-pending-requests 32 --pending-timeout 600s `
  --host 127.0.0.1 --port 18092 --model-id qwen27-dspine `
  --chat-template tools/chat_templates/qwen3_8.jinja
```

The [benchmark guide](https://github.com/Dylan997S/ninfer-5090-windows/blob/v1.2.5-dspine.3/bench/dspine-5090.md)
reproduces the 32-request Python test with a 2,304-token context and 2,048 output tokens
per request. Linux matched-workload results are documented in the Linux guide.

The v1.2.5-dspine.5 packages also support encrypted reasoning replay, automatic history
truncation and process-local background Responses. See the [Responses compatibility guide](https://github.com/Dylan997S/ninfer-dspine-rtx5090/blob/v1.2.5-dspine.5/docs/responses-compatibility.md).

## Convert a model / 转换模型

Use a Python 3.12 environment with PyTorch, NumPy, safetensors and huggingface_hub.
The DSpine companion and lookup-table conversion run on CPU. If you already have the
v1.2.5 `qwen3_8_27b_nvfp4full_noex.v3.ninfer` target, reuse it as the `--base`.
To build that target from the port's pinned checkpoints, run `python build_model.py --line noex --dest models --hf-src hf-src`
from the extracted folder. That optional base build requires CUDA-enabled PyTorch.
Leave room for the downloads and both output files. The final model is about 19.8 GiB.

Obtain `config.json` and `model.safetensors` from
[tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/tree/ea909c0f5fab8865f9e14600334f55f7ce47308f)
at revision `ea909c0f5fab8865f9e14600334f55f7ce47308f`, in a directory named `draft-hf`.
The `hf` CLI is provided by huggingface_hub.

```powershell
hf download tencent/Qwen3.8-27B-DSpine-b16 config.json model.safetensors --revision ea909c0f5fab8865f9e14600334f55f7ce47308f --local-dir draft-hf
python -m tools.convert.dspine --base models/qwen3_8_27b_nvfp4full_noex.v3.ninfer --draft draft-hf --out models/qwen27-dspine-nvfp4.v3.ninfer --projection-format nvfp4
python -m tools.convert.dspine_tables --base models/qwen27-dspine-nvfp4.v3.ninfer --out models/qwen27-dspine-nvfp4-tables.v3.ninfer
python -m tools.convert.verify_dspine_tables --base models/qwen27-dspine-nvfp4.v3.ninfer --tables models/qwen27-dspine-nvfp4-tables.v3.ninfer --out tables-check.json
.\start_dspine_nvfp4.bat
```

The converter verifies the copied target objects by SHA-256. The table checker samples both
BF16 lookup tables against an FP64 reference. The ZIP contains the engine and converters.
Model files are downloaded or built separately.

## Checksums and source

`SHA256SUMS` records the hashes of the extracted files. The matching GitHub Release publishes a
separate SHA-256 for the entire ZIP. Source and benchmark methodology are in the
[NVFP4 DSpine branch](https://github.com/Dylan997S/ninfer-5090-windows/tree/windows/nvfp4-dspine).
