# /v1/responses compatibility

Version v1.2.5-dspine.5 adds the following Responses API behavior to the Windows and Linux RTX 5090 packages. Model conversion and DSpine launch profiles follow the existing release recipes.

## Added behavior

- `include: ["reasoning.encrypted_content"]` returns an authenticated AES-256-GCM envelope on reasoning output items. Clients can replay that item with its plaintext `content` omitted, including with `store: false`. Modified envelopes and mismatched plaintext are rejected.
- `truncation: "auto"` measures the rendered prompt and removes the oldest complete conversation rounds until the input and explicitly requested output fit the configured context. System/developer messages and the latest user round are retained. Function calls and their results stay together. A latest round that cannot fit returns `context_length_exceeded`. Stored continuations use the truncated history.
- `background: true` returns a queued Response while generation runs independently of the submitting connection. `GET /v1/responses/{id}` polls the result. `POST /v1/responses/{id}/cancel` cancels queued or active work and is idempotent. `DELETE /v1/responses/{id}` cancels active work before removing it.
- `background: true, stream: true` emits SSE and retains events. `GET /v1/responses/{id}?stream=true&starting_after=N` resumes after an event's sequence number. Disconnecting an SSE client leaves background inference running.

## Local lifetime and capacity

Encrypted reasoning is readable by the same running server process. Its random key is held in memory and changes on restart. Cloud-generated encrypted envelopes cannot be decrypted by this local server.

Responses and retained SSE events are process-local. The background registry is bounded by the configured response-store record/byte limits. With the default 256 MiB budget, it holds up to 16 background jobs, with a 16 MiB reservation each. Input/initial metadata and response/event buffers each receive half a reservation. Pending work is never evicted. Completed jobs can be evicted when another job needs a slot. Exceeding a buffer produces a terminal error. `store: false` still retains the bounded background job temporarily for polling and cancellation.

Foreground execution and explicit `truncation: "disabled"` are supported. Linux builds use OpenSSL 3 via the `libssl-dev` build dependency and `libssl3t64` runtime on Ubuntu 24.04. Windows uses the system BCrypt API.

## Reproduce the package checks

The repository includes `tests/integration/responses_compat.py`. Install `openai==2.20.0` in a test environment, then pass `--package`, `--model` and `--output` paths. The script checks the extracted package's SHA256SUMS, starts its launcher and runs the same request suite on Windows and Linux. It exercises ordinary requests, SSE, tool returns, the three repaired options, cancellation, error polling and stream recovery. The selected loopback port must be available and the GPU must have enough free memory for the model.

Linux release validation is scoped to Ubuntu 24.04 under WSL2. Bare-metal Linux remains untested. Other clients can be checked with the same local endpoint.

## Verified packages

On 2026-11-12, Windows 11 Pro build 26200 and Ubuntu 24.04 WSL2 each passed the same
28 real-model HTTP/SDK checks and three host-side unit suites. The GPU was one RTX 5090.
The model, template, default single-active-request profile and test inputs matched across platforms.
OpenAI Python SDK 2.20.0 was used on both systems. The Linux build used GCC 13.3,
CUDA 13.4 and OpenSSL 3.0.13. Windows used MSVC 19.44 and CUDA 13.3.33.

The checks cover ordinary and typed requests, SSE framing and deltas, stateful and stateless
tool returns, encrypted-only reasoning replay and tamper rejection, automatic truncation and
input-token counting, background polling, queued/active cancellation, disconnect recovery,
terminal errors and deletion. Linux shut down cleanly after the suite.

These checks validate Responses compatibility with the default 8,192-token, one-active-request
launcher. Historical 32-request throughput measurements are available in the preceding release.
Native Linux and container execution were outside this validation.
