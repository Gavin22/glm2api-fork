"""Pull cumulative text snapshots from a debug dump, raw upstream frames only.

The log interleaves dumps from several stages (raw block, parsed payload,
computed delta, finalize output). Extracting every "text" field mixes them
together and produces a misleading sequence, so this reads only the raw
upstream frames and reports how each one relates to the one before.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BACKSLASH = chr(92)
QUOTED = '"(?:[^"' + BACKSLASH + BACKSLASH + ']|' + BACKSLASH + BACKSLASH + '.)*"'
TEXT_FIELD = re.compile('"text":' + BACKSLASH + 's*(' + QUOTED + ')')
RAW_FRAME = re.compile(r"GLM 原始 SSE block\s*\n\s*(.*)")


def snapshots_from_raw_frames(raw: str) -> list[str]:
    frames = RAW_FRAME.findall(raw)
    seen: list[str] = []
    for frame in frames:
        for match in TEXT_FIELD.findall(frame):
            try:
                value = json.loads(match)
            except json.JSONDecodeError:
                continue
            seen.append(value)
    return seen


def classify(previous: str, current: str) -> str:
    if previous == current:
        return "identical"
    if current.startswith(previous):
        return f"append (+{len(current) - len(previous)})"
    if previous.startswith(current):
        return f"REWIND ({len(previous)} -> {len(current)})"
    return "MUTATED"


def main() -> int:
    raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    snaps = snapshots_from_raw_frames(raw)

    print(f"{len(snaps)} text snapshots across raw upstream frames")
    for index, value in enumerate(snaps):
        print(f"[{index:>3}] len={len(value):>6}  {value[:70]!r}")

    print()
    print("--- relation to previous ---")
    counts: dict[str, int] = {}
    for index in range(1, len(snaps)):
        kind = classify(snaps[index - 1], snaps[index])
        key = kind.split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
        print(f"  {index - 1:>3} -> {index:>3}  {kind}")
    print()
    print("summary:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())