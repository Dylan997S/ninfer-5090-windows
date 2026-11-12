"""Run real-model Responses regressions against an extracted portable package.

Requires Python 3.10+ and openai==2.20.0. The package server is started on its
default loopback port and stopped after the checks. Keep the GPU idle and stop
other inference services before running this integration suite.
"""
import argparse
import hashlib
import signal
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", required=True, type=Path)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--base-url", default="http://127.0.0.1:18092")
options = parser.parse_args()
PACKAGE = options.package.resolve()
MODEL = options.model.resolve()
OUT = options.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
API = options.base_url.rstrip("/")
TOOL = {"type": "function", "name": "add_numbers", "description": "Add two integers.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"], "additionalProperties": False}, "strict": True}
QUESTION = "Use add_numbers to calculate 19 plus 23. After the tool returns, reply with only its numeric result."


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def body(text, **extra):
    return {"model": "qwen27-dspine", "input": text, "max_output_tokens": 256,
            "temperature": 0, "reasoning": {"effort": "none"}, **extra}


def request(path, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=90)


def text_of(response):
    return "".join(part.get("text", "") for item in response.get("output", [])
                   if item.get("type") == "message" for part in item.get("content", [])
                   if part.get("type") == "output_text")


def call(label, payload, stream=False):
    record = {"request": payload}
    try:
        with request("/v1/responses", payload) as response:
            record["http_status"] = response.status
            record["content_type"] = response.headers.get("Content-Type")
            if stream:
                raw = response.read().decode()
                record["sse_raw"] = raw
                events = []
                for block in raw.replace("\r\n", "\n").split("\n\n"):
                    name, data = None, []
                    for line in block.splitlines():
                        if line.startswith("event:"):
                            name = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].lstrip())
                    if data:
                        events.append({"event": name, "data": json.loads("\n".join(data))})
                record["events"] = events
                terminal = [event["data"] for event in events
                            if event["data"].get("type") in ("response.completed", "response.incomplete", "response.failed")]
                record["response"] = terminal[-1].get("response") if terminal else None
            else:
                record["response"] = json.load(response)
    except urllib.error.HTTPError as error:
        record.update(http_status=error.code, error_body=error.read().decode())
    except Exception as error:
        record["exception"] = repr(error)
    save(label + ".json", record)
    return record


def ordinary(record, expected):
    value = record.get("response") or {}
    assert record.get("http_status") == 200, record
    assert value.get("object") == "response" and value.get("status") == "completed", value
    assert value.get("id") and value.get("usage", {}).get("output_tokens", 0) > 0, value
    assert text_of(value).strip() == expected, value


def stream_check(record):
    events = record["events"]
    names = [e["data"]["type"] for e in events]
    assert names[0:2] == ["response.created", "response.in_progress"], names
    assert names.count("response.completed") == 1 and names[-1] == "response.completed", names
    sequence = [e["data"]["sequence_number"] for e in events]
    assert sequence == sorted(set(sequence)), sequence
    assert all(e["event"] == e["data"]["type"] for e in events), events
    deltas = "".join(e["data"].get("delta", "") for e in events if e["data"]["type"] == "response.output_text.delta")
    assert deltas == text_of(record["response"]), (deltas, record["response"])


def tool_check(record, streamed=False):
    value = record.get("response") or {}
    assert record.get("http_status") == 200 and value.get("status") == "completed", record
    calls = [item for item in value["output"] if item["type"] == "function_call"]
    assert len(calls) == 1, value
    item = calls[0]
    assert item["name"] == "add_numbers" and json.loads(item["arguments"]) == {"a": 19, "b": 23}, item
    assert item["id"] != item["call_id"], item
    if streamed:
        stream_check(record)
        deltas = "".join(e["data"].get("delta", "") for e in record["events"]
                         if e["data"]["type"] == "response.function_call_arguments.delta")
        assert deltas == item["arguments"], deltas
    return item


def get_json(path, payload=None, method=None):
    with request(path, payload, method) as response:
        return json.load(response)


def check_error(label, payload, code):
    result = call(label, payload)
    assert result.get("http_status") == 400, result
    assert json.loads(result["error_body"])["error"]["code"] == code, result


def await_background(response, label):
    deadline = time.monotonic() + 90
    seen = [response]
    while response["status"] in ("queued", "in_progress"):
        assert time.monotonic() < deadline, response
        time.sleep(.05)
        response = get_json("/v1/responses/" + response["id"])
        seen.append(response)
    save(label + "-polls.json", seen)
    return response


def run_compat_tests(test):
    def sdk_roundtrip():
        from openai import OpenAI
        client = OpenAI(base_url=API + "/v1", api_key="local", timeout=90, max_retries=0)
        result = client.responses.create(model="qwen27-dspine", input="Reply only SDK_OK.",
            max_output_tokens=128, reasoning={"effort":"none"},
            include=["reasoning.encrypted_content"], truncation="auto")
        assert result.output_text.strip() == "SDK_OK", result
        with client.responses.stream(model="qwen27-dspine", input="Reply only SDK_STREAM_OK.",
            max_output_tokens=128, reasoning={"effort":"none"}) as stream:
            stream.until_done()
            final = stream.get_final_response()
        assert final.output_text.strip() == "SDK_STREAM_OK", final
        initial = client.responses.create(model="qwen27-dspine", input="Reply only SDK_BG_OK.",
            max_output_tokens=128, reasoning={"effort":"none"}, background=True)
        final = initial
        deadline = time.monotonic() + 60
        while final.status in ("queued", "in_progress"):
            assert time.monotonic() < deadline
            time.sleep(.05)
            final = client.responses.retrieve(initial.id)
        assert final.output_text.strip() == "SDK_BG_OK", final
        save("sdk.json", {"ordinary":result.model_dump(), "background":final.model_dump()})
        client.close()
    test("openai_sdk_ordinary_stream_background", sdk_roundtrip)

    def include_plain():
        result = call("include_plain", body("Reply only INCLUDE_OK.", include=["reasoning.encrypted_content"]))
        ordinary(result, "INCLUDE_OK")
    test("include_plain", include_plain)

    def encrypted_roundtrip(stream):
        label = "encrypted_stream" if stream else "encrypted"
        result = call(label, body(QUESTION, reasoning={"effort": "low"}, tools=[TOOL],
            tool_choice={"type": "function", "name": "add_numbers"}, stream=stream,
            store=False, include=["reasoning.encrypted_content"]), stream)
        tool = tool_check(result, stream)
        output = result["response"]["output"]
        reasoning = [item for item in output if item["type"] == "reasoning"]
        assert reasoning and reasoning[0]["encrypted_content"].startswith("ninfer-gcm-v1."), output
        if stream:
            done = [e["data"]["item"] for e in result["events"]
                    if e["event"] == "response.output_item.done" and e["data"]["item"]["type"] == "reasoning"]
            assert done[0]["encrypted_content"] == reasoning[0]["encrypted_content"], done
        replay = json.loads(json.dumps(output))
        for item in replay:
            if item["type"] == "reasoning":
                item.pop("content", None)
        items = [{"role": "user", "content": QUESTION}, *replay,
                 {"type": "function_call_output", "call_id": tool["call_id"], "output": '{"result":42}'}]
        ordinary(call(label + "_return", body(items, tools=[TOOL], store=False)), "42")
        cipher = next(item for item in replay if item["type"] == "reasoning")
        cipher["encrypted_content"] = cipher["encrypted_content"][:-1] + ("0" if cipher["encrypted_content"][-1] != "0" else "1")
        check_error(label + "_tamper", body(items, tools=[TOOL]), "invalid_encrypted_content")
    test("encrypted_stateless_tool_roundtrip", lambda: encrypted_roundtrip(False))
    test("encrypted_stream_tool_roundtrip", lambda: encrypted_roundtrip(True))

    def retrieve_encrypted():
        result = call("retrieve_encrypted_create", body("Calculate 6 * 7, then reply only 42.", reasoning={"effort":"low"}))
        ordinary(result, "42")
        value = get_json("/v1/responses/" + result["response"]["id"] + "?include%5B%5D=reasoning.encrypted_content")
        assert any(item.get("encrypted_content") for item in value["output"] if item["type"] == "reasoning"), value
        save("retrieve_encrypted.json", value)
    test("retrieve_encrypted", retrieve_encrypted)

    old_text = "obsolete history filler " * 4500
    history = [{"role":"user", "content": old_text}, {"role":"assistant", "content":"Old answer."},
               {"role":"user", "content":"Reply only TRIM_OK."}]
    test("truncation_disabled_rejects_long_input", lambda: check_error("truncation_disabled", body(history), "context_length_exceeded"))

    def truncate():
        counted = get_json("/v1/responses/input_tokens", {
            "model": "qwen27-dspine", "input": history, "truncation": "auto"})
        assert counted["input_tokens"] < 512, counted
        save("truncation_count.json", counted)
        result = call("truncation_auto", body(history, truncation="auto", instructions="Always follow the last user's exact output instruction."))
        ordinary(result, "TRIM_OK")
        assert result["response"]["truncation"] == "auto" and result["response"]["usage"]["input_tokens"] < 512
        follow = call("truncation_followup", body("Reply only STILL_OK.", previous_response_id=result["response"]["id"]))
        ordinary(follow, "STILL_OK")
        assert follow["response"]["usage"]["input_tokens"] < 512, follow
    test("truncation_auto_and_stored_followup", truncate)

    def truncate_tool_group():
        items = [history[0], {"type":"function_call", "call_id":"call_old", "name":"add_numbers", "arguments":'{"a":1,"b":2}'},
                 {"type":"function_call_output", "call_id":"call_old", "output":"3"}, history[-1]]
        ordinary(call("truncate_tool_group", body(items, tools=[TOOL], truncation="auto")), "TRIM_OK")
    test("truncation_preserves_tool_group_integrity", truncate_tool_group)
    test("truncation_latest_turn_too_large", lambda: check_error("truncation_latest_too_large", body(old_text, truncation="auto"), "context_length_exceeded"))

    def background_plain():
        started = time.monotonic()
        initial = call("background_plain", body("Reply only BG_OK.", background=True, store=False))["response"]
        assert initial["status"] == "queued" and initial["background"] is True, initial
        final = await_background(initial, "background_plain")
        assert final["status"] == "completed" and text_of(final).strip() == "BG_OK", final
        save("background_plain-timing.json", {"elapsed_seconds":time.monotonic()-started})
    test("background_plain_poll_store_false", background_plain)

    def background_tool():
        result = call("background_tool", body(QUESTION, background=True, tools=[TOOL],
            tool_choice={"type":"function", "name":"add_numbers"}))["response"]
        final = await_background(result, "background_tool")
        tool = tool_check({"http_status":200,"response":final})
        value = body([{"type":"function_call_output","call_id":tool["call_id"],"output":"42"}],
                     previous_response_id=final["id"], tools=[TOOL], background=True)
        follow = await_background(call("background_tool_return", value)["response"], "background_tool_return")
        assert follow["status"] == "completed" and text_of(follow).strip() == "42", follow
    test("background_tool_return", background_tool)

    def background_cancel():
        result = call("background_cancel_create", body("Write a long Python module with many functions and detailed docstrings. Continue for at least 3000 words.",
            background=True, max_output_tokens=2048))["response"]
        cancelled = get_json("/v1/responses/" + result["id"] + "/cancel", {}, "POST")
        assert cancelled["status"] == "cancelled", cancelled
        again = get_json("/v1/responses/" + result["id"] + "/cancel", {}, "POST")
        assert again == cancelled, (again, cancelled)
        fetched = get_json("/v1/responses/" + result["id"])
        assert fetched == cancelled
        save("background_cancel.json", cancelled)
    test("background_cancel_idempotent", background_cancel)

    def queued_cancel():
        active = call("background_active_for_queue", body(
            "Write a comprehensive Python networking library with hundreds of functions. Keep generating code.",
            background=True, max_output_tokens=2048))["response"]
        try:
            pending = call("background_queued_for_cancel", body("Reply only QUEUED.", background=True))["response"]
            started = time.monotonic()
            value = get_json("/v1/responses/" + pending["id"] + "/cancel", {}, "POST")
            elapsed = time.monotonic() - started
            assert value["status"] == "cancelled" and elapsed < 2, (value, elapsed)
            save("background_queue_cancel.json", {"elapsed_seconds":elapsed,"response":value})
        finally:
            get_json("/v1/responses/" + active["id"] + "/cancel", {}, "POST")
    test("background_queued_cancel", queued_cancel)

    def background_stream():
        result = call("background_stream", body("Reply only BG_STREAM_OK.", background=True, stream=True), True)
        stream_check(result)
        ordinary(result, "BG_STREAM_OK")
        assert result["events"][0]["data"]["response"]["status"] == "queued"
    test("background_stream", background_stream)

    def background_disconnect_resume():
        payload = body("Write 100 numbered lines with a useful Python programming tip on each line.",
                       background=True, stream=True, max_output_tokens=1024)
        with request("/v1/responses", payload) as response:
            initial = None
            for line in response:
                if line.startswith(b"data: "):
                    event = json.loads(line[6:])
                    initial = event["response"]
                    seq = event["sequence_number"]
                    break
        assert initial is not None
        final = await_background(initial, "background_disconnected")
        assert final["status"] in ("completed", "incomplete"), final
        with request("/v1/responses/" + initial["id"] + "?stream=true&starting_after=" + str(seq)) as response:
            raw = response.read().decode()
        data = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
        assert data[0]["sequence_number"] == seq + 1
        assert all(e["sequence_number"] > seq for e in data)
        assert data[-1]["response"] == final
        save("background_resumed.json", {"initial_sequence":seq, "events":data})
    test("background_disconnect_and_resume", background_disconnect_resume)

    def background_failure():
        initial = call("background_failure", body("too large " * 6000, background=True))["response"]
        final = await_background(initial, "background_failure")
        assert final["status"] == "failed" and final["error"]["code"] == "context_length_exceeded", final
    test("background_failure_poll", background_failure)

    def background_delete():
        initial = call("background_delete_create", body("Reply only DELETE_OK.", background=True, store=False))["response"]
        await_background(initial, "background_delete")
        result = get_json("/v1/responses/" + initial["id"], method="DELETE")
        assert result["deleted"] is True
        try:
            get_json("/v1/responses/" + initial["id"])
            raise AssertionError("deleted response remained accessible")
        except urllib.error.HTTPError as error:
            assert error.code == 404
    test("background_delete", background_delete)


def main():
    hashes = {}
    for line in (PACKAGE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected_hash, name = line.split("  ", 1)
        with (PACKAGE / name).open("rb") as file:
            hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
        assert hashes[name] == expected_hash, f"Package file changed: {name}"
    env = dict(os.environ)
    if os.name == "nt":
        executable = PACKAGE / "ninfer-serve.exe"
        env["PATH"] = os.pathsep.join([str(Path(os.environ["WINDIR"]) / "System32"), os.environ["WINDIR"]])
        args = [os.environ["COMSPEC"], "/d", "/c", "call", str(PACKAGE / "start_dspine_nvfp4.bat"), str(MODEL)]
        process_options = {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        executable = PACKAGE / "apps/ninfer-serve"
        args = [str(PACKAGE / "start_dspine_nvfp4.sh"), str(MODEL)]
        process_options = {"start_new_session": True}
    save("provenance.json", {"exe": str(executable), "hashes": hashes, "model": str(MODEL),
                             "platform": os.name, "args": args})
    results = {}

    def test(label, operation):
        try:
            operation()
            results[label] = {"passed": True}
        except Exception as error:
            results[label] = {"passed": False, "error": str(error)}
        save("summary.json", results)
        print(label, "PASS" if results[label]["passed"] else "FAIL", flush=True)

    with (OUT / "stdout.log").open("wb") as stdout, (OUT / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(args, cwd=PACKAGE, env=env, stdout=stdout, stderr=stderr,
                                 **process_options)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"Portable server exited {child.returncode}")
                try:
                    with request("/health") as response:
                        if json.load(response)["status"] == "ok":
                            break
                except OSError:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("Portable server readiness timed out")

            test("plain", lambda: ordinary(call("plain", body("Reply only READY.")), "READY"))
            test("typed_input", lambda: ordinary(call("typed_input", body(
                [{"role": "user", "content": [{"type": "input_text", "text": "Reply only TYPED."}]}])), "TYPED"))

            def streamed():
                result = call("stream", body("Reply only STREAM_OK.", stream=True), True)
                ordinary(result, "STREAM_OK")
                stream_check(result)
            test("stream", streamed)
            roots = {}
            for use_stream in (False, True):
                label = "tool_stream" if use_stream else "tool"
                def initial_tool():
                    result = call(label, body(QUESTION, tools=[TOOL],
                        tool_choice={"type": "function", "name": "add_numbers"}, stream=use_stream), use_stream)
                    item = tool_check(result, use_stream)
                    roots[label] = (result["response"], item)
                test(label, initial_tool)

            for label, (response, item) in roots.items():
                output = {"type": "function_call_output", "call_id": item["call_id"],
                          "output": json.dumps({"result": 42})}
                test(label + "_stateful_return", lambda: ordinary(call(label + "_stateful_return",
                    body([output], previous_response_id=response["id"], tools=[TOOL], tool_choice="auto")), "42"))
                replay = [{"role": "user", "content": QUESTION}, *response["output"], output]
                test(label + "_stateless_return", lambda: ordinary(call(label + "_stateless_return",
                    body(replay, tools=[TOOL], tool_choice="auto", store=False)), "42"))
                def retrieve():
                    with request("/v1/responses/" + response["id"]) as raw:
                        value = json.load(raw)
                    assert value["id"] == response["id"] and value["output"] == response["output"]
                test(label + "_retrieve", retrieve)

            run_compat_tests(test)
        finally:
            if child.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill.exe", "/T", "/F", "/PID", str(child.pid)], capture_output=True)
                else:
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)

    if any(not result.get("passed", False) for result in results.values()):
        raise RuntimeError("Compatibility regression failed, see summary.json")

if __name__ == "__main__":
    main()
