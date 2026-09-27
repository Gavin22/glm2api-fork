"""Reproduce the Claude-Code-breaking defects in the DSML tool bridge.

Run:  python -X utf8 tests/repro_defects.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from glm2api.utils.tool_parser import StreamingToolParser, parse_tool_calls_from_text  # noqa: E402
from glm2api.utils.tool_protocol import (  # noqa: E402
    serialize_tool_call_block,
    serialize_tool_result_block,
)

ALLOWED = {"Read", "Write", "Edit", "Bash", "Grep"}


def roundtrip(name: str, args: dict) -> dict:
    block = serialize_tool_call_block(name, json.dumps(args, ensure_ascii=False))
    _, calls = parse_tool_calls_from_text(block, allowed_tool_names=ALLOWED)
    if not calls:
        return {"__lost__": True}
    return json.loads(calls[0]["function"]["arguments"])


def roundtrip_via_translator(name: str, args: dict) -> dict:
    """Snapshot/restore through the translator, which is what upstream sees."""
    from glm2api.services.translator import convert_messages

    block = serialize_tool_call_block(name, json.dumps(args, ensure_ascii=False))
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{name}", "name": name, "content": block},
    ]
    prompt = convert_messages(messages, tools=None)[0]["content"][0]["text"]
    # The assistant turn is re-serialized into DSML; parse it back out.
    _, calls = parse_tool_calls_from_text(prompt, allowed_tool_names=None)
    if not calls:
        return {"__lost__": True}
    return json.loads(calls[0]["function"]["arguments"])


def check(label: str, ok: bool) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


print("=" * 78)
print("1. Write/Edit content fidelity  (the edit-will-never-match defect)")
print("=" * 78)
cases = [
    ("Edit.old_string", "Edit", {"file_path": "/tmp/x.py", "old_string": "def f():\n    pass\n", "new_string": "def f():\n    return 1\n"}),
    ("Write.content", "Write", {"file_path": "/tmp/a.py", "content": "import os\n\n\ndef main():\n\tprint('hi')\n"}),
    ("Bash.command", "Bash", {"command": "grep -rn 'foo' src/ | head -20"}),
]
for label, tool, args in cases:
    back = roundtrip(tool, args)
    same = back == args
    check(f"{label} round-trips exactly", same)
    if not same:
        print(f"        sent: {args}")
        print(f"        back: {back}")

print()
print("=" * 78)
print("2. Content containing the protocol's own tags")
print("=" * 78)
hostile = [
    ("close-parameter", {"file_path": "/p.py", "content": "docs: </|DSML|parameter> end\n"}),
    ("close-tool_calls", {"file_path": "/p.py", "content": 'x = "</|DSML|tool_calls>"\n'}),
    ("cdata-end", {"file_path": "/p.py", "content": "if a ]]> b:\n    pass\n"}),
    ("nested-write", {"file_path": "/p.py", "content": "write <|DSML|invoke name=\"x\"> here\n"}),
]
for label, args in hostile:
    back = roundtrip("Write", args)
    ok = back == args
    check(f"{label}: preserved", ok)
    if not ok:
        print(f"        sent: {args['content']!r}")
        print(f"        back: {back.get('content', back)!r}")

print()
print("=" * 78)
print("3. Empty containers and pure-whitespace values")
print("=" * 78)
for label, args in [
    ("empty list", {"items": []}),
    ("empty dict", {"meta": {}}),
    ("whitespace-only", {"old_string": "\t\n  "}),
]:
    back = roundtrip("Bash", args)
    check(f"{label}: preserved", back == args)
    if back != args:
        print(f"        sent: {args}")
        print(f"        back: {back}")

print()
print("=" * 78)
print("4. Truncation at max_tokens (silent empty turn)")
print("=" * 78)
truncated = '<|DSML|tool_calls>\n  <|DSML|invoke name="Bash">\n    <|DSML|parameter name="command">ls -la'
parser = StreamingToolParser(allowed_tool_names=ALLOWED)
visible = parser.consume(truncated)
tail, calls = parser.flush()
check("truncated block yields a usable tool call", len(calls) == 1)
check("truncated block does not vanish silently", bool(visible + tail) or bool(calls))
print(f"        visible={visible!r} tail={tail!r} calls={len(calls)}")

print()
print("=" * 78)
print("5. Model drifts to JSON instead of DSML")
print("=" * 78)
drifting = [
    '```json\n{"name": "Bash", "arguments": {"command": "pwd"}}\n```',
    'Bash({"command": "pwd"})',
    '<tool_call>{"tool": "Bash", "params": {"command": "pwd"}}</tool_call>',
]
for text in drifting:
    _, calls = parse_tool_calls_from_text(text, allowed_tool_names=ALLOWED)
    check(f"recovered JSON drift: {text[:44]}...", len(calls) == 1)

print()
print("=" * 78)
print("6. Undeclared tool name (silent empty assistant message)")
print("=" * 78)
ghost = '<|DSML|tool_calls>\n  <|DSML|invoke name="ghost_tool">\n    <|DSML|parameter name="a">1</|DSML|parameter>\n  </|DSML|invoke>\n</|DSML|tool_calls>'
clean, calls = parse_tool_calls_from_text(ghost, allowed_tool_names=ALLOWED)
check("undeclared name is reported, not silently swallowed", bool(clean) or bool(calls))
print(f"        clean={clean!r} calls={len(calls)}")

print()
print("=" * 78)
print("7. Tool result body escaping")
print("=" * 78)
body = "file contents with </|DSML|tool_result> inside"
out = serialize_tool_result_block("call_1", "Read", body)
check("result body close-tag is escaped", "</|DSML|tool_result" not in out.split("<content>", 1)[1].rsplit("</content>", 1)[0])

print()
print("=" * 78)
print("8. Parser throughput on a large Write argument")
print("=" * 78)
import time  # noqa: E402

big = "x" * 100_000
block = serialize_tool_call_block("Write", json.dumps({"file_path": "/big.txt", "content": big}))
parser = StreamingToolParser(allowed_tool_names=ALLOWED)
start = time.perf_counter()
for i in range(0, len(block), 100):
    parser.consume(block[i : i + 100])
parser.flush()
elapsed = time.perf_counter() - start
check(f"100KB argument parsed under 1.0s (took {elapsed:.2f}s)", elapsed < 1.0)
