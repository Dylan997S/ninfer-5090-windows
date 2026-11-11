@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL=%ROOT%models\qwen27-dspine-nvfp4-tables.v3.ninfer"
if not "%~1"=="" set "MODEL=%~f1"
if not exist "%MODEL%" (
    echo Model not found: "%MODEL%"
    echo See START-HERE.md for conversion instructions, or pass the model path.
    exit /b 2
)
set "NINFER_CUDA_SYNC=blocking"
"%ROOT%ninfer-serve.exe" "%MODEL%" --spec dspine --draft-tokens 11 --proposal-head full --kv-dtype fp8 --max-context 8192 --context-capacity 8GiB --host-cache-capacity 8GiB --max-concurrency 1 --prefill-chunk 1024 --no-prefix-reuse --host 127.0.0.1 --port 18092 --model-id qwen27-dspine --chat-template "%ROOT%tools\chat_templates\qwen3_8.jinja"
exit /b %ERRORLEVEL%
