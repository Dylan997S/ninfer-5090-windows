"""Reproduce the RTX 5090 DSpine 32-request throughput workloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time
import urllib.request


FIXTURES = Path(__file__).resolve().parent / "fixtures"
WIDTH = 32
OUTPUT_TOKENS = 2048


def width32_steps(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        body = response.read().decode()
    match = re.search(r'^ninfer_decode_step_seconds_count\{batch="32"\}\s+(\d+)\s*$', body, re.M)
    if match is None:
        raise RuntimeError("Width-32 decode counter missing")
    return int(match[1])


def generate(base, model, case, barrier):
    body = {"model": model, "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": OUTPUT_TOKENS, "temperature": 0, "seed": 42,
            "presence_penalty": 1.5, "frequency_penalty": 0, "ignore_eos": False,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    usage, finish = None, None
    barrier.wait()
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[5:])
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
    if (not usage or usage["prompt_tokens"] != case["input_tokens"] or
            usage["completion_tokens"] != OUTPUT_TOKENS or finish != "length"):
        raise RuntimeError(f"Unexpected token counts or stop reason for {case['topic']}: {usage}, {finish}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18092")
    parser.add_argument("--model", default="qwen27-dspine")
    parser.add_argument("--workload", choices=("diverse", "acceptance"), default="diverse")
    parser.add_argument("--rounds", type=int)
    args = parser.parse_args()
    if args.workload == "diverse":
        cases = json.loads((FIXTURES / "dspine-diverse32.json").read_text(encoding="utf-8"))["cases"]
    else:
        prompts = json.loads((FIXTURES / "dspine-workload.json").read_text(encoding="utf-8"))
        cases = [{"prompt": prompts["groups"]["python_code"][index % 2],
                  "input_tokens": (110, 104)[index % 2], "topic": f"code-{index}"}
                 for index in range(WIDTH)]
    rounds = args.rounds if args.rounds is not None else (5 if args.workload == "diverse" else 1)
    if len(cases) != WIDTH or rounds < 1:
        parser.error("Expected 32 cases and at least one round")

    elapsed = 0.0
    with ThreadPoolExecutor(max_workers=WIDTH) as pool:
        for round_number in range(1, rounds + 1):
            before = width32_steps(args.base)
            barrier = threading.Barrier(WIDTH + 1)
            futures = [pool.submit(generate, args.base, args.model, case, barrier) for case in cases]
            start = time.perf_counter()
            barrier.wait()
            for future in futures:
                future.result()
            seconds = time.perf_counter() - start
            steps = width32_steps(args.base) - before
            if steps < 1:
                raise RuntimeError("The engine recorded no width-32 decode steps")
            elapsed += seconds
            print(f"Round {round_number}: 32/32, {WIDTH * OUTPUT_TOKENS / seconds:.1f} tok/s, "
                  f"{steps} width-32 steps", flush=True)
    print(f"Pooled: {rounds * WIDTH * OUTPUT_TOKENS / elapsed:.1f} tok/s across {rounds} rounds")


if __name__ == "__main__":
    main()
