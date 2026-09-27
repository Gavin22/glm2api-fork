"""End-to-end test of the Claude Code path through the real HTTP server.

A fake upstream stands in for chatglm.cn so the whole chain is exercised:
Claude Code's SSE on one side, the GLM web protocol on the other, with the
real server, client, translator and parser in between.

Run:  python -X utf8 tests/e2e_claude_code.py
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")

from glm2api.config import AppConfig  # noqa: E402
from glm2api.logging_utils import get_logger, setup_logging  # noqa: E402
from glm2api.server import GLM2APIServer  # noqa: E402
from glm2api.services.glm_client import GLMWebClient  # noqa: E402

API_KEY = "sk-e2e-test-key"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def glm_events(text_parts: list[str], conversation_id: str = "conv_e2e_1"):
    """Mimic the GLM stream: cumulative snapshots, then a finish status."""
    events = [{"conversation_id": conversation_id, "status": "init", "parts": []}]
    for index, text in enumerate(text_parts):
        events.append(
            {
                "conversation_id": conversation_id,
                "status": "generating",
                "parts": [
                    {"logic_id": "1", "role": "assistant", "status": "generating", "content": [{"type": "text", "text": text}]}
                ],
            }
        )
    events.append(
        {
            "conversation_id": conversation_id,
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "role": "assistant",
                    "status": "finish",
                    "content": [{"type": "text", "text": text_parts[-1] if text_parts else ""}],
                }
            ],
        }
    )
    return events


class FakeUpstream(BaseHTTPRequestHandler):
    """Stands in for https://chatglm.cn/chatglm."""

    scenario: str = "tool_call"
    seen_bodies: list[dict] = []

    def log_message(self, *args) -> None:  # silence
        return

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            parsed = {}

        if self.path.endswith("/user-api/user/refresh") or self.path.endswith("/user-api/guest/access"):
            return self._json(
                {"code": 0, "result": {"access_token": "fake-access-token", "refresh_token": "fake-refresh-token"}}
            )

        if self.path.endswith("/conversation/delete"):
            return self._json({"code": 0})

        if self.path.endswith("/assistant/stream"):
            FakeUpstream.seen_bodies.append(parsed)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in self._scenario_events():
                self.wfile.write(sse(event))
                self.wfile.flush()
                time.sleep(0.001)
            return

        return self._json({"code": 0})

    def _scenario_events(self) -> list[dict]:
        if FakeUpstream.scenario == "tool_call":
            # The model narrates, then emits a Write call whose content
            # contains indentation, a trailing newline, and a line that looks
            # like the proxy's own protocol. All three used to corrupt it.
            body = "line one\n\tline two </|DSML|tool_calls> tail\n"
            call = (
                "<|DSML|tool_calls><|DSML|invoke name=\"Write\">"
                f"<|DSML|parameter name=\"file_path\">/tmp/out.txt</|DSML|parameter>"
                f"<|DSML|parameter name=\"content\">{body}</|DSML|parameter>"
                "</|DSML|invoke></|DSML|tool_calls>"
            )
            return glm_events([f"Writing the file now.\n{call}"])
        if FakeUpstream.scenario == "truncated":
            # Cut off mid-argument, the way max_tokens truncation does.
            return glm_events(['<|DSML|tool_calls><|DSML|invoke name="Bash"><|DSML|parameter name="command">ls -la'])
        if FakeUpstream.scenario == "json_drift":
            return glm_events(['```json\n{"name": "Bash", "arguments": {"command": "pwd"}}\n```'])
        if FakeUpstream.scenario == "undeclared":
            return glm_events(['<|DSML|tool_calls><|DSML|invoke name="ghost_tool"><|DSML|parameter name="a">1</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'])
        return glm_events(["Hello from GLM."])


def build_config(upstream_port: int, api_port: int) -> AppConfig:
    return AppConfig(
        env_file_path=None,  # type: ignore[arg-type]
        env_file_created=False,
        token_file_path=None,  # type: ignore[arg-type]
        host="127.0.0.1",
        port=api_port,
        api_prefix="/v1",
        log_level="WARNING",
        debug_dump_all=False,
        request_timeout=30,
        glm_base_url=f"http://127.0.0.1:{upstream_port}/chatglm",
        glm_use_guest_refresh_token=False,
        glm_refresh_token="fake-refresh-token",
        glm_refresh_tokens=["fake-refresh-token"],
        glm_assistant_id="65940acff94777010aa6b796",
        glm_image_assistant_id="65a232c082ff90a2ad2f15e2",
        glm_image_model_name="glm-image-1",
        glm_user_agent="e2e-test-agent",
        glm_delete_conversation=False,
        glm_max_concurrency=2,
        glm_queue_wait_timeout=30,
        glm_busy_max_retries=0,
        glm_busy_retry_interval=0.0,
        glm_guest_max_retries=0,
        blocked_tool_names=set(),
        exposed_models=["glm-5.2"],  # type: ignore[arg-type]
        model_aliases={"glm-5.2": "glm-5.2"},
        server_api_keys=[API_KEY],
        cors_allow_origin="*",
    )


def anthropic_request(payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{API_PORT}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_stream_events() -> list[dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{API_PORT}/v1/messages",
        data=json.dumps(
            {
                "model": "glm-5.2",
                "max_tokens": 1024,
                "stream": True,
                "tools": TOOLS,
                "messages": [{"role": "user", "content": "write the file"}],
            }
        ).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=30) as response:
        event_name = ""
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                events.append({"event": event_name, "data": json.loads(line[6:])})
    return events


TOOLS = [
    {
        "name": "Write",
        "description": "Write a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a command",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
]

PASSED = 0
FAILED = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"[PASS] {label}")
    else:
        FAILED += 1
        print(f"[FAIL] {label}")
        if detail:
            print(f"        {detail}")


def tool_use_from(events: list[dict]) -> tuple[dict | None, str]:
    """Assemble the tool_use block the way a client would."""
    blocks: dict[int, dict] = {}
    partial: dict[int, str] = {}
    text = ""
    stop_reason = ""
    for item in events:
        data = item["data"]
        if item["event"] == "content_block_start" and data["content_block"]["type"] == "tool_use":
            blocks[data["index"]] = data["content_block"]
            partial[data["index"]] = ""
        elif item["event"] == "content_block_delta":
            delta = data["delta"]
            if delta["type"] == "input_json_delta":
                partial[data["index"]] = partial.get(data["index"], "") + delta["partial_json"]
            elif delta["type"] == "text_delta":
                text += delta["text"]
        elif item["event"] == "message_delta":
            stop_reason = data["delta"].get("stop_reason") or stop_reason
    for index, block in blocks.items():
        try:
            block["input"] = json.loads(partial.get(index, "") or "{}")
        except json.JSONDecodeError:
            block["input"] = {"__unparseable__": partial.get(index, "")}
    return (next(iter(blocks.values())) if blocks else None), text


UPSTREAM_PORT = free_port()
API_PORT = free_port()

setup_logging("ERROR")
upstream = ThreadingHTTPServer(("127.0.0.1", UPSTREAM_PORT), FakeUpstream)
threading.Thread(target=upstream.serve_forever, daemon=True).start()

config = build_config(UPSTREAM_PORT, API_PORT)
client = GLMWebClient(config=config, logger=get_logger("e2e.glm"))
api = GLM2APIServer(config=config, glm_client=client, logger=get_logger("e2e.http"))
threading.Thread(target=api.serve_forever, daemon=True).start()
time.sleep(0.2)

try:
    print("=" * 78)
    print("1. Claude Code tool_use round trip (Write with hostile content)")
    print("=" * 78)
    FakeUpstream.scenario = "tool_call"
    events = collect_stream_events()
    block, text = tool_use_from(events)

    check("a tool_use block reached the client", block is not None)
    check("stop_reason is tool_use", any(
        item["event"] == "message_delta" and item["data"]["delta"].get("stop_reason") == "tool_use" for item in events
    ))
    if block:
        expected_content = "line one\n\tline two </|DSML|tool_calls> tail\n"
        check("tool name preserved", block["name"] == "Write", str(block))
        check("file_path preserved", block["input"].get("file_path") == "/tmp/out.txt", str(block["input"]))
        check(
            "content preserved byte-for-byte (newlines, tab, protocol-looking text)",
            block["input"].get("content") == expected_content,
            f"got {block['input'].get('content')!r}",
        )
        # Upstream deliberately withholds prose on a turn that produced a tool
        # call, to stop models narrating tool choices at clients that would
        # render it as an answer. Locked in by
        # test_accumulator_drops_tool_preamble_and_repairs_shell_command_array.
        check("narration is withheld on a tool-call turn (upstream design)", text.strip() == "", repr(text))

    print()
    print("=" * 78)
    print("2. Truncated block at max_tokens becomes a usable call")
    print("=" * 78)
    FakeUpstream.scenario = "truncated"
    events = collect_stream_events()
    block, text = tool_use_from(events)
    check("truncated block still produced a tool_use", block is not None, repr(text))
    if block:
        check("its arguments survived the truncation", block["input"].get("command") == "ls -la", str(block["input"]))

    print()
    print("=" * 78)
    print("3. JSON drift is recovered, not shown as prose")
    print("=" * 78)
    FakeUpstream.scenario = "json_drift"
    events = collect_stream_events()
    block, text = tool_use_from(events)
    check("drifted JSON became a tool_use", block is not None, repr(text))
    if block:
        check("drifted call keeps its arguments", block["input"].get("command") == "pwd", str(block["input"]))
    check("raw JSON was not emitted as assistant text", '"name"' not in text, repr(text))

    print()
    print("=" * 78)
    print("4. Undeclared tool reported, never an empty turn")
    print("=" * 78)
    FakeUpstream.scenario = "undeclared"
    events = collect_stream_events()
    block, text = tool_use_from(events)
    check("no fabricated tool_use for an undeclared name", block is None)
    check("the client was told what happened", "ghost_tool" in text, repr(text))
    check("turn is not empty", text.strip() != "")

    print()
    print("=" * 78)
    print("5. Plain conversation still works")
    print("=" * 78)
    FakeUpstream.scenario = "plain"
    events = collect_stream_events()
    block, text = tool_use_from(events)
    check("no tool call for a plain answer", block is None)
    check("text delivered", "Hello from GLM" in text, repr(text))

    print()
    print("=" * 78)
    print("6. Upstream request shape")
    print("=" * 78)
    body = FakeUpstream.seen_bodies[-1] if FakeUpstream.seen_bodies else {}
    check("tools were serialized into the prompt", "# TOOL SCHEMAS" in json.dumps(body, ensure_ascii=False))
    check("no native tools field was sent upstream", "tools" not in body)

finally:
    upstream.shutdown()
    api.shutdown()

print()
print("=" * 78)
print(f"total: {PASSED} passed, {FAILED} failed")
print("=" * 78)
sys.exit(1 if FAILED else 0)
