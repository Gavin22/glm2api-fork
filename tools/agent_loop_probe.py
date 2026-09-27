"""Drive a multi-turn agent loop and measure real tool-call reliability.

Mimics what Claude Code does: a small tool set, a multi-turn loop, real
local tool execution, and a report of every turn's outcome. Pointed at a
live endpoint it measures the actual DSML adherence rate; pointed at the
fake upstream it validates the harness itself.

Usage:
  python -X utf8 tools/agent_loop_probe.py --base-url http://127.0.0.1:8000 --key sk-... --turns 12
  python -X utf8 tools/agent_loop_probe.py --mock          # self-check, no network

Reported per turn: whether a tool_use arrived, whether its arguments were
parseable and schema-valid, whether the named tool was declared, and the
failure mode when it was not.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

TOOLS = [
    {
        "name": "Read",
        "description": "Read a file from disk and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Absolute path"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating it if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "Grep",
        "description": "Search files for a pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]

SCHEMA_REQUIRED = {
    name: tool["input_schema"]["required"] for name, tool in ((t["name"], t) for t in TOOLS)
}

# A multi-turn task that needs several different tools in sequence, the way a
# real coding session does.
TASK_PROMPTS = [
    "Create a file {root}/greeting.py containing a function greet(name) that returns 'Hello, ' + name.",
    "Read {root}/greeting.py back and tell me how many lines it has.",
    "Append a __main__ block to {root}/greeting.py that prints greet('world').",
    "Run python on {root}/greeting.py and show me the output.",
    "Search the file for the word 'def' and report every match.",
    "Create {root}/notes.md summarising what the script does.",
    "Use Grep to find 'greet' in {root} and list the files.",
    "Read {root}/notes.md and confirm it exists.",
    "Write a file {root}/empty.txt with exactly three blank lines.",
    "Read {root}/empty.txt and report its byte size via Bash.",
    "Create {root}/nested/deep/file.txt with the text 'deep'.",
    "Search {root} for 'deep' and report where it appears.",
]


class Stats:
    def __init__(self) -> None:
        self.turns = 0
        self.tool_use_turns = 0
        self.text_only_turns = 0
        self.empty_turns = 0
        self.parse_failures = 0
        self.schema_failures = 0
        self.undeclared_tool = 0
        self.executed_ok = 0
        self.executed_error = 0
        self.failure_modes: Counter[str] = Counter()
        self.raw_samples: list[dict] = []

    def report(self) -> str:
        lines = [
            "",
            "=" * 78,
            "AGENT LOOP RESULT",
            "=" * 78,
            f"turns run                  {self.turns}",
            f"turns with a tool_use      {self.tool_use_turns}  ({self._pct(self.tool_use_turns)})",
            f"turns text-only (no call)  {self.text_only_turns}  ({self._pct(self.text_only_turns)})",
            f"turns completely empty     {self.empty_turns}  ({self._pct(self.empty_turns)})",
            "",
            f"arguments unparseable      {self.parse_failures}  ({self._pct(self.parse_failures)})",
            f"required arg missing       {self.schema_failures}  ({self._pct(self.schema_failures)})",
            f"undeclared tool name       {self.undeclared_tool}  ({self._pct(self.undeclared_tool)})",
            "",
            f"tool executed cleanly      {self.executed_ok}  ({self._pct(self.executed_ok)})",
            f"tool returned an error     {self.executed_error}  ({self._pct(self.executed_error)})",
        ]
        if self.turns:
            usable = self.executed_ok
            lines.append("")
            lines.append(f"END-TO-END USABLE CALLS    {usable}/{self.turns}  ({100.0 * usable / self.turns:.1f}%)")
        if self.failure_modes:
            lines += ["", "failure modes:"]
            for mode, count in self.failure_modes.most_common():
                lines.append(f"  {count:>3}x  {mode}")
        return "\n".join(lines)

    def _pct(self, value: int) -> str:
        if not self.turns:
            return "n/a"
        return f"{100.0 * value / self.turns:.1f}%"


def post_messages(base_url: str, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Accept": "text/event-stream",
        },
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=600) as response:
        event_name = ""
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                try:
                    events.append({"event": event_name, "data": json.loads(line[6:])})
                except json.JSONDecodeError:
                    continue
    return assemble(events)


def assemble(events: list[dict]) -> dict:
    """Rebuild the assistant turn from the SSE stream."""
    blocks: dict[int, dict] = {}
    partial: dict[int, str] = {}
    text = ""
    stop_reason = None
    for item in events:
        data = item["data"]
        kind = item["event"]
        if kind == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                blocks[data["index"]] = dict(block)
                partial[data["index"]] = ""
            elif block.get("type") == "text":
                text += block.get("text", "")
        elif kind == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "input_json_delta":
                partial[data["index"]] = partial.get(data["index"], "") + delta.get("partial_json", "")
            elif delta.get("type") == "text_delta":
                text += delta.get("text", "")
        elif kind == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason", stop_reason)
    parsed: list[dict] = []
    for index, block in blocks.items():
        raw = partial.get(index, "")
        try:
            block["input"] = json.loads(raw) if raw.strip() else {}
            block["_parse_error"] = None
        except json.JSONDecodeError as exc:
            block["input"] = {}
            block["_parse_error"] = str(exc)
        parsed.append(block)
    return {"text": text, "tool_uses": parsed, "stop_reason": stop_reason}


# --------------------------------------------------------------------------
# Local tool execution
# --------------------------------------------------------------------------


def run_tool(name: str, args: dict, root: str) -> tuple[bool, str]:
    try:
        if name == "Read":
            with open(args["file_path"], "r", encoding="utf-8") as handle:
                return True, handle.read()[:4000]
        if name == "Write":
            path = args["file_path"]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(args["content"])
            return True, f"wrote {len(args['content'])} chars"
        if name == "Bash":
            command = args["command"]
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30, cwd=root
            )
            return result.returncode == 0, (result.stdout + result.stderr)[:2000]
        if name == "Grep":
            matches: list[str] = []
            for dirpath, _dirs, files in os.walk(root):
                for filename in files:
                    full = os.path.join(dirpath, filename)
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as handle:
                            for lineno, line in enumerate(handle, 1):
                                if args["pattern"] in line:
                                    matches.append(f"{full}:{lineno}: {line.rstrip()[:120]}")
                    except OSError:
                        continue
            return True, "\n".join(matches[:40]) or "(no matches)"
    except subprocess.TimeoutExpired:
        return False, "command timed out"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except KeyError as exc:
        return False, f"missing argument {exc}"
    return False, f"unknown tool {name}"


def classify_turn(stats: Stats, turn: dict, declared: set[str], root: str) -> tuple[str, bool]:
    """Record one turn. Returns (failure mode or "", executed cleanly)."""
    uses = turn["tool_uses"]
    if not uses:
        if turn["text"].strip():
            stats.text_only_turns += 1
            return "model answered with text instead of calling a tool", False
        stats.empty_turns += 1
        return "EMPTY TURN (no text, no tool call)", False

    stats.tool_use_turns += 1
    all_ok = True
    for use in uses:
        name = use.get("name", "")
        if use.get("_parse_error"):
            stats.parse_failures += 1
            stats.failure_modes["arguments were not valid JSON"] += 1
            all_ok = False
            continue
        if name not in declared:
            stats.undeclared_tool += 1
            stats.failure_modes[f"undeclared tool name: {name!r}"] += 1
            all_ok = False
            continue
        required = SCHEMA_REQUIRED.get(name, [])
        missing = [field for field in required if field not in use["input"]]
        if missing:
            stats.schema_failures += 1
            stats.failure_modes[f"{name} called without required {missing}"] += 1
            all_ok = False
            continue
    if not all_ok:
        return "tool call rejected before execution", False

    executed = True
    for use in uses:
        ok, output = run_tool(use["name"], use["input"], root)
        if ok:
            stats.executed_ok += 1
        else:
            stats.executed_error += 1
            stats.failure_modes[f"{use['name']} errored: {output[:60]}"] += 1
            executed = False
    return ("", executed)


def build_history(turns: list[tuple[str, dict, str]]) -> list[dict]:
    """Replay the conversation the way Claude Code does."""
    messages: list[dict] = []
    for prompt, turn, result_text in turns:
        messages.append({"role": "user", "content": prompt})
        content: list[dict] = []
        if turn["text"].strip():
            content.append({"type": "text", "text": turn["text"]})
        for use in turn["tool_uses"]:
            content.append(
                {
                    "type": "tool_use",
                    "id": use.get("id", "toolu_probe"),
                    "name": use.get("name", ""),
                    "input": use.get("input", {}),
                }
            )
        if not content:
            content.append({"type": "text", "text": "(no output)"})
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": use.get("id", "toolu_probe"),
                        "content": result_text,
                    }
                    for use in turn["tool_uses"]
                ]
                or [{"type": "text", "text": result_text}],
            }
        )
    return messages


def run_loop(base_url: str, api_key: str, model: str, turns: int, verbose: bool) -> Stats:
    stats = Stats()
    root = tempfile.mkdtemp(prefix="glm_probe_")
    declared = set(SCHEMA_REQUIRED)
    history: list[tuple[str, dict, str]] = []

    for index in range(turns):
        prompt = TASK_PROMPTS[index % len(TASK_PROMPTS)].format(root=root)
        payload = {
            "model": model,
            "max_tokens": 4096,
            "stream": True,
            "tools": TOOLS,
            "messages": build_history(history) + [{"role": "user", "content": prompt}],
        }
        started = time.time()
        try:
            turn = post_messages(base_url, api_key, payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            stats.turns += 1
            stats.failure_modes[f"HTTP {exc.code}: {body[:80]}"] += 1
            print(f"turn {index + 1:>2}: HTTP {exc.code} {body}")
            break
        except Exception as exc:  # noqa: BLE001
            stats.turns += 1
            stats.failure_modes[f"{type(exc).__name__}: {exc}"] += 1
            print(f"turn {index + 1:>2}: {type(exc).__name__}: {exc}")
            break

        stats.turns += 1
        mode, executed = classify_turn(stats, turn, declared, root)
        calls = ", ".join(use.get("name", "?") for use in turn["tool_uses"]) or "-"
        flag = "ok " if executed and not mode else "!! "
        print(
            f"turn {index + 1:>2}: {flag}stop={turn['stop_reason'] or '-':<10} tools={calls:<24} "
            f"text={len(turn['text']):>5}ch {time.time() - started:5.1f}s"
        )
        if verbose and mode:
            print(f"          mode: {mode}")
            print(f"          text: {turn['text'][:300]!r}")
            stats.raw_samples.append({"prompt": prompt, "text": turn["text"], "uses": turn["tool_uses"]})

        result_text = ""
        for use in turn["tool_uses"]:
            ok, output = run_tool(use["name"], use["input"], root) if not mode else (False, mode)
            result_text += output + "\n"
        if not turn["tool_uses"]:
            result_text = "You did not call a tool. Please use one of the available tools."
        history.append((prompt, turn, result_text[:4000]))

    return stats


# --------------------------------------------------------------------------
# Self-check against the fake upstream
# --------------------------------------------------------------------------


class MockUpstream(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path.endswith("/user-api/user/refresh") or self.path.endswith("/user-api/guest/access"):
            body = json.dumps({"code": 0, "result": {"access_token": "k", "refresh_token": "k"}}).encode()
        elif self.path.endswith("/assistant/stream"):
            body = None
        else:
            body = json.dumps({"code": 0}).encode()
        if body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        script = (
            "<|DSML|tool_calls><|DSML|invoke name=\"Write\">"
            "<|DSML|parameter name=\"file_path\">%s/greeting.py</|DSML|parameter>"
            "<|DSML|parameter name=\"content\">def greet(name):\n    return 'Hello, ' + name\n</|DSML|parameter>"
            "</|DSML|invoke></|DSML|tool_calls>"
        )
        payload = {
            "conversation_id": "c",
            "status": "finish",
            "parts": [{"logic_id": "1", "role": "assistant", "status": "finish", "content": [{"type": "text", "text": script}]}],
        }
        self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def mock_self_check() -> int:
    """Prove the harness measures what it claims, using a cooperating model."""
    from glm2api.config import AppConfig
    from glm2api.logging_utils import get_logger, setup_logging
    from glm2api.server import GLM2APIServer
    from glm2api.services.glm_client import GLMWebClient

    upstream_port, api_port = free_port(), free_port()
    setup_logging("ERROR")
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), MockUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    config = AppConfig(
        env_file_path=None, env_file_created=False, token_file_path=None,  # type: ignore[arg-type]
        host="127.0.0.1", port=api_port, api_prefix="/v1", log_level="ERROR", debug_dump_all=False,
        request_timeout=30, glm_base_url=f"http://127.0.0.1:{upstream_port}/chatglm",
        glm_use_guest_refresh_token=False, glm_refresh_token="k", glm_refresh_tokens=["k"],
        glm_assistant_id="65940acff94777010aa6b796", glm_image_assistant_id="65a232c082ff90a2ad2f15e2",
        glm_image_model_name="glm-image-1", glm_user_agent="probe", glm_delete_conversation=False,
        glm_max_concurrency=2, glm_queue_wait_timeout=30, glm_busy_max_retries=0, glm_busy_retry_interval=0.0,
        glm_guest_max_retries=0, blocked_tool_names=set(), exposed_models=["glm-5.2"],  # type: ignore[arg-type]
        model_aliases={"glm-5.2": "glm-5.2"}, server_api_keys=["sk-probe"], cors_allow_origin="*",
    )
    client = GLMWebClient(config=config, logger=get_logger("probe.glm"))
    api = GLM2APIServer(config=config, glm_client=client, logger=get_logger("probe.http"))
    threading.Thread(target=api.serve_forever, daemon=True).start()
    time.sleep(0.2)
    try:
        stats = run_loop(f"http://127.0.0.1:{api_port}", "sk-probe", "glm-5.2", 3, verbose=True)
        print(stats.report())
        ok = stats.tool_use_turns == 3 and stats.executed_ok == 3 and not stats.parse_failures
        print()
        print("[PASS] harness self-check: a cooperating model scores 3/3" if ok else "[FAIL] harness self-check")
        return 0 if ok else 1
    finally:
        upstream.shutdown()
        api.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default=os.environ.get("GLM2API_KEY", ""))
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--mock", action="store_true", help="self-check against a cooperating fake model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mock:
        return mock_self_check()
    if not args.key:
        parser.error("--key (or GLM2API_KEY) is required unless --mock is used")

    print(f"driving {args.turns} turns against {args.base_url} model={args.model}")
    print(f"tools: {', '.join(SCHEMA_REQUIRED)}")
    stats = run_loop(args.base_url, args.key, args.model, args.turns, args.verbose)
    print(stats.report())
    if args.verbose and stats.raw_samples:
        with open("agent_probe_samples.json", "w", encoding="utf-8") as handle:
            json.dump(stats.raw_samples, handle, ensure_ascii=False, indent=2)
        print("\nraw failure samples written to agent_probe_samples.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
