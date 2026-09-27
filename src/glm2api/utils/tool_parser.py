from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .tool_protocol import BLOCKED_NATIVE_TOOL_NAMES

CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
TOOL_RESULT_PATTERN = re.compile(
    r"<(?:(?:\|DSML\|)|ml_)?tool_result\b[\s\S]*?</(?:(?:\|DSML\|)|ml_)?tool_result>",
    re.IGNORECASE,
)
START_TAG_PATTERN = re.compile(
    r"<(?P<tag>\|DSML\|tool_calls|tool_calls|ml_tool_calls|ml_tool_call)\b[^>]*>",
    re.IGNORECASE,
)
DSML_TAG_PATTERN = re.compile(r"</?\|DSML\|(?P<name>tool_calls|invoke|parameter|tool_result)\b", re.IGNORECASE)
DSML_OPEN_TAG_PATTERN = re.compile(
    r"<\|dsml\|(?P<name>tool_calls|toolcalls|invoke|parameter|tool_result|toolresult)\b(?P<attrs>[^<>]*?)>",
    re.IGNORECASE,
)
DSML_CLOSE_TAG_PATTERN = re.compile(
    r"</\|dsml\|(?P<name>tool_calls|toolcalls|invoke|parameter|tool_result|toolresult)\s*\|?\s*>",
    re.IGNORECASE,
)
DSML_COMPACT_CLOSE_TAG_PATTERN = re.compile(
    r"(?:</\|dsml|<\|/dsml)(?P<name>toolcalls|invoke|parameter|toolresult)\s*\|\s*>",
    re.IGNORECASE,
)
DSML_DOUBLE_PIPE_CLOSE_TAG_PATTERN = re.compile(
    r"<\|\|dsml\|(?P<name>tool_calls|toolcalls|invoke|parameter|tool_result|toolresult)\s*\|?\s*>",
    re.IGNORECASE,
)
DSML_TOOL_CALLS_CLOSE_PATTERN = re.compile(
    r"(?:</\|dsml\|tool_calls\s*>|</\|dsml\|tool_calls\s*\|\s*>|</\|dsmltool_?calls\s*\|\s*>|<\|/dsmltool_?calls\s*\|\s*>)",
    re.IGNORECASE,
)
DSML_TOOL_CALLS_TRAILING_CLOSE_PATTERN = re.compile(
    r"(?:</\|dsml\|tool_calls\s*>|</\|dsml\|tool_calls\s*\|\s*>|</\|dsmltool_?calls\s*\|\s*>|<\|/dsmltool_?calls\s*\|\s*>|</\|dsml\|tool_calls\s*$|</\|dsmltool_?calls\s*\|?\s*$|<\|/dsmltool_?calls\s*\|?\s*$)",
    re.IGNORECASE,
)
PARAM_NAME_TAG_PATTERN = re.compile(r"<param_name>\s*(.*?)\s*</param_name>", re.IGNORECASE | re.DOTALL)
PARAM_VALUE_TAG_PATTERN = re.compile(r"<param_value>\s*(.*?)\s*</param_value>", re.IGNORECASE | re.DOTALL)
TAG_NAME_HINTS = [
    "<|",
    "</|",
    "<|DSML|",
    "</|DSML|",
    "<|DSML|tool_calls",
    "</|DSML|tool_calls",
    "<|DSML|invoke",
    "</|DSML|invoke",
    "<|DSML|parameter",
    "</|DSML|parameter",
    "<|DSML|tool_result",
    "</|DSML|tool_result",
    "<m",
    "</m",
    "<ml_",
    "</ml_",
    "<ml_tool_calls",
    "</ml_tool_calls",
    "<ml_tool_call",
    "</ml_tool_call",
    "<ml_tool_name",
    "</ml_tool_name",
    "<ml_parameters",
    "</ml_parameters",
    "<ml_tool_result",
    "</ml_tool_result",
    "<tool_calls",
    "</tool_calls",
    "<invoke",
    "</invoke",
    "<parameter",
    "</parameter",
]


def _local_name(tag: str) -> str:
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    return tag.lower()


def _canonical_dsml_name(name: str) -> str:
    normalized = name.lower().replace("_", "")
    if normalized == "toolcalls":
        return "tool_calls"
    if normalized == "toolresult":
        return "tool_result"
    return normalized


def _repair_malformed_dsml(block: str) -> str:
    if "<|" not in block and "]]|>" not in block:
        return block

    repaired = block.replace("]]|>", "]]>")
    if "<![CDATA[" in repaired:
        repaired = re.sub(
            r"(?<!\])\]>(?=</\|dsml\|parameter\b|</\|DSML\|parameter\b|</parameter\b|</\|dsmlparameter\|)",
            "]]>",
            repaired,
            flags=re.IGNORECASE,
        )

    def replace_open(match: re.Match[str]) -> str:
        name = _canonical_dsml_name(match.group("name"))
        attrs = match.group("attrs").rstrip("|").rstrip()
        return f"<|DSML|{name}{attrs}>"

    def replace_close(match: re.Match[str]) -> str:
        return f"</|DSML|{_canonical_dsml_name(match.group('name'))}>"

    repaired = DSML_OPEN_TAG_PATTERN.sub(replace_open, repaired)
    repaired = DSML_CLOSE_TAG_PATTERN.sub(replace_close, repaired)
    repaired = DSML_COMPACT_CLOSE_TAG_PATTERN.sub(replace_close, repaired)
    repaired = DSML_DOUBLE_PIPE_CLOSE_TAG_PATTERN.sub(replace_close, repaired)
    repaired = re.sub(
        r"(?:</\|dsml\|tool_calls|</\|dsmltool_?calls|<\|/dsmltool_?calls)\s*\|?\s*$",
        "</|DSML|tool_calls>",
        repaired,
        flags=re.IGNORECASE,
    )
    return repaired


def _normalize_dsml_to_xml(block: str) -> str:
    repaired = _repair_malformed_dsml(block)
    return DSML_TAG_PATTERN.sub(lambda match: match.group(0).replace("|DSML|", ""), repaired)


def _is_allowed_tool_name(tool_name: str, allowed_tool_names: set[str] | None) -> bool:
    if tool_name in BLOCKED_NATIVE_TOOL_NAMES:
        return False
    return allowed_tool_names is None or tool_name in allowed_tool_names


def _leaf_text(element: ET.Element) -> str:
    """Raw text of a leaf element.

    Newlines, tabs and runs of spaces are load-bearing: they carry the exact
    bytes of ``Write.content`` / ``Edit.old_string`` / ``Edit.new_string``.
    Collapsing them here is what made every file edit fail to match on disk,
    so this must return the text verbatim.
    """
    return "".join(element.itertext())


def _coerce_leaf_value(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            if stripped.startswith("[") and not stripped.endswith("]"):
                try:
                    return json.loads(stripped + "]")
                except json.JSONDecodeError:
                    pass
            return text
    if stripped in {"true", "false"}:
        return stripped == "true"
    if stripped == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        try:
            return float(stripped)
        except ValueError:
            return text
    return text


def _append_value(mapping: dict[str, object], key: str, value: object) -> None:
    if key not in mapping:
        mapping[key] = value
        return
    existing = mapping[key]
    if isinstance(existing, list):
        existing.append(value)
        return
    mapping[key] = [existing, value]


def _xml_value_to_object(element: ET.Element) -> object:
    children = [child for child in list(element) if isinstance(child.tag, str)]
    if not children:
        # ``type="obj"`` is emitted by the serializer for an empty object,
        # which is otherwise indistinguishable from an empty string.
        if element.attrib.get("type") == "obj":
            return {}
        # ``<parameter name="items"></parameter>`` is an empty list, not "".
        if _is_empty_container(element):
            return [] if _names_a_list(element) else ""
        return _coerce_leaf_value(_leaf_text(element))

    repeated_item_only = all(_local_name(child.tag) == "item" for child in children)
    if repeated_item_only:
        return [_xml_value_to_object(child) for child in children]

    result: dict[str, object] = {}
    for child in children:
        key = child.attrib.get("name", "").strip() or _local_name(child.tag)
        _append_value(result, key, _xml_value_to_object(child))
    return result


def _is_empty_container(element: ET.Element) -> bool:
    """True for ``<x></x>`` / ``<x/>``: no children and no text at all.

    Wholly-whitespace text is *not* empty — it is a real payload for
    file-editing tools, which is why the raw text is checked rather than
    ``strip()``-ed.
    """
    if list(element):
        return False
    if element.attrib.get("type") == "obj":
        return False
    return "".join(element.itertext()) == ""


def _names_a_list(element: ET.Element) -> bool:
    name = (element.attrib.get("name", "") or _local_name(element.tag)).strip().lower()
    return name.endswith("s") or name in {"list", "array", "items"}


def _extract_tool_name(element: ET.Element) -> str:
    if _local_name(element.tag) == "invoke":
        return element.attrib.get("name", "").strip()
    for tag_name in ("ml_tool_name", "tool_name"):
        tool_name_element = element.find(tag_name)
        if tool_name_element is not None:
            return _leaf_text(tool_name_element)
    return ""


def _extract_arguments(element: ET.Element) -> dict[str, object] | None:
    if _local_name(element.tag) == "invoke":
        parameters: dict[str, object] = {}
        parameter_children = [
            child
            for child in list(element)
            if isinstance(child.tag, str) and _local_name(child.tag) == "parameter"
        ]
        for child in parameter_children:
            key = child.attrib.get("name", "").strip()
            if key:
                _append_value(parameters, key, _xml_value_to_object(child))
        if parameter_children:
            return parameters
        # Models sometimes wrap a JSON object body in the invoke instead of
        # emitting parameter tags. Treating that as "no arguments" produced a
        # tool_use with empty input, so read the body instead.
        body = (element.text or "").strip()
        if body.startswith("{"):
            try:
                parsed_body = json.loads(body)
            except json.JSONDecodeError:
                return parameters
            if isinstance(parsed_body, dict):
                return parsed_body
        return parameters

    for tag_name in ("ml_parameters", "parameters"):
        parameters_element = element.find(tag_name)
        if parameters_element is not None:
            parsed = _xml_value_to_object(parameters_element)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
    return None


def _build_tool_call(name: str, arguments: dict[str, object], index: int) -> dict[str, object]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "index": index,
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def _parse_tool_call_element(
    element: ET.Element,
    allowed_tool_names: set[str] | None,
    index: int,
) -> dict[str, object] | None:
    if _local_name(element.tag) not in {"invoke", "tool_call", "ml_tool_call"}:
        return None

    tool_name = _extract_tool_name(element)
    if not tool_name:
        return None
    if not _is_allowed_tool_name(tool_name, allowed_tool_names):
        return None

    arguments = _extract_arguments(element)
    if arguments is None:
        return None

    return _build_tool_call(tool_name, arguments, index)


def _extract_malformed_tool_call_from_root(
    root: ET.Element,
    allowed_tool_names: set[str] | None,
    index: int,
) -> dict[str, object] | None:
    root_name = _local_name(root.tag)
    if root_name not in {"tool_calls", "ml_tool_calls"}:
        return None

    tool_name = _extract_tool_name(root)
    if not tool_name:
        return None
    if not _is_allowed_tool_name(tool_name, allowed_tool_names):
        return None

    for tag_name in ("ml_parameters", "parameters"):
        parameters_element = root.find(tag_name)
        if parameters_element is not None:
            parsed = _xml_value_to_object(parameters_element)
            arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
            return _build_tool_call(tool_name, arguments, index)

    names = [match.group(1).strip() for match in PARAM_NAME_TAG_PATTERN.finditer(ET.tostring(root, encoding="unicode"))]
    values = [match.group(1).strip() for match in PARAM_VALUE_TAG_PATTERN.finditer(ET.tostring(root, encoding="unicode"))]
    if names and values and len(names) == len(values):
        arguments = {
            key: _coerce_leaf_value(value)
            for key, value in zip(names, values, strict=False)
            if key
        }
        return _build_tool_call(tool_name, arguments, index)
    if names and not values:
        return None

    direct_pairs: dict[str, object] = {}
    children = [child for child in list(root) if isinstance(child.tag, str)]
    for child in children:
        key = _local_name(child.tag)
        if key in {"tool_name", "ml_tool_name", "tool_call", "ml_tool_call"}:
            continue
        if key in {"param_name", "param_value"}:
            continue
        direct_pairs[key] = _xml_value_to_object(child)
    if direct_pairs:
        return _build_tool_call(tool_name, direct_pairs, index)
    return None


def _parse_xml_block(
    block: str,
    allowed_tool_names: set[str] | None,
    start_index: int,
) -> tuple[list[dict[str, object]], tuple[int, int] | None]:
    try:
        root = ET.fromstring(_normalize_dsml_to_xml(block))
    except ET.ParseError:
        return [], None

    root_name = _local_name(root.tag)
    if root_name in {"tool_calls", "ml_tool_calls"}:
        candidates = [
            child
            for child in list(root)
            if isinstance(child.tag, str) and _local_name(child.tag) in {"invoke", "tool_call", "ml_tool_call"}
        ]
    elif root_name in {"tool_call", "ml_tool_call"}:
        candidates = [root]
    else:
        return [], None

    tool_calls: list[dict[str, object]] = []
    for candidate in candidates:
        parsed = _parse_tool_call_element(candidate, allowed_tool_names, len(tool_calls))
        if parsed is not None:
            tool_calls.append(parsed)

    if not tool_calls:
        malformed = _extract_malformed_tool_call_from_root(root, allowed_tool_names, 0)
        if malformed is not None:
            tool_calls.append(malformed)

    if not tool_calls:
        return [], None
    return tool_calls, (start_index, start_index + len(block))


def _mask_code_fences(text: str) -> str:
    """Blank out fenced code so markup inside it is not scanned as a call."""
    matches = list(CODE_FENCE_PATTERN.finditer(text))
    if not matches:
        # No fence: return the original. Copying every character into a list
        # and joining it back was the single hottest cost when streaming a
        # large tool argument.
        return text
    masked = list(text)
    for match in matches:
        for index in range(match.start(), match.end()):
            masked[index] = " "
    return "".join(masked)


def _find_matching_block(
    masked_text: str,
    start_match: re.Match[str],
    *,
    allow_trailing_close: bool = False,
) -> tuple[int, int] | None:
    tag_name = start_match.group("tag").lower()
    if tag_name == "|dsml|tool_calls":
        closing_pattern = DSML_TOOL_CALLS_TRAILING_CLOSE_PATTERN if allow_trailing_close else DSML_TOOL_CALLS_CLOSE_PATTERN
    else:
        closing_pattern = re.compile(rf"</{re.escape(tag_name)}\s*>", re.IGNORECASE)
    closing_match = closing_pattern.search(masked_text, start_match.end())
    if closing_match is None:
        return None
    return start_match.start(), closing_match.end()


def _is_empty_args_call(
    tool_call: dict[str, object],
    required_params: dict[str, list[str]] | None,
) -> bool:
    """True when a call declares required parameters but carries none.

    The model sometimes emits well-formed DSML whose body is a JSON object
    instead of parameter tags, which parses to ``{}``. Delivering that to the
    client means running e.g. ``Bash`` with no ``command``: the tool errors, the
    error is fed back, and the loop spins. Rejecting it lets the caller report
    the failure instead.
    """
    if not required_params:
        return False
    function = tool_call.get("function", {})
    if not isinstance(function, dict):
        return False
    required = required_params.get(str(function.get("name", "")))
    if not required:
        return False
    arguments = str(function.get("arguments", ""))
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return False
    return parsed == {} or parsed is None


def _extract_tool_blocks(
    text: str,
    allowed_tool_names: set[str] | None,
    *,
    allow_trailing_close: bool = False,
    required_params: dict[str, list[str]] | None = None,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    masked_text = _mask_code_fences(text)
    spans: list[tuple[int, int]] = []
    tool_calls: list[dict[str, object]] = []
    cursor = 0

    while cursor < len(masked_text):
        match = START_TAG_PATTERN.search(masked_text, cursor)
        if match is None:
            break
        span = _find_matching_block(masked_text, match, allow_trailing_close=allow_trailing_close)
        if span is None:
            break

        start, end = span
        block_calls, parsed_span = _parse_xml_block(text[start:end], allowed_tool_names, start)
        block_calls = [call for call in block_calls if not _is_empty_args_call(call, required_params)]
        if parsed_span is not None and block_calls:
            for offset, tool_call in enumerate(block_calls, start=len(tool_calls)):
                tool_call["index"] = offset
            spans.append(parsed_span)
            tool_calls.extend(block_calls)
            cursor = end
            continue
        if match.group("tag").lower() in {"|dsml|tool_calls", "tool_calls", "ml_tool_calls", "ml_tool_call"}:
            spans.append((start, end))
            cursor = end
            continue

        cursor = match.end()

    return spans, tool_calls


def _remove_spans(text: str, spans: list[tuple[int, int]], *, trim_outer_whitespace: bool = True) -> str:
    if not spans:
        cleaned = TOOL_RESULT_PATTERN.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip() if trim_outer_whitespace else cleaned

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    cleaned = "".join(parts)
    cleaned = TOOL_RESULT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() if trim_outer_whitespace else cleaned


def _find_unmatched_fence_start(text: str) -> int | None:
    last_open = None
    cursor = 0
    while True:
        index = text.find("```", cursor)
        if index == -1:
            break
        if last_open is None:
            last_open = index
        else:
            last_open = None
        cursor = index + 3
    return last_open


def _find_incomplete_block_start(text: str, *, allow_trailing_close: bool = False) -> int | None:
    masked_text = _mask_code_fences(text)
    cursor = 0
    while cursor < len(masked_text):
        match = START_TAG_PATTERN.search(masked_text, cursor)
        if match is None:
            break
        span = _find_matching_block(masked_text, match, allow_trailing_close=allow_trailing_close)
        if span is None:
            return match.start()
        cursor = span[1]
    return None


def _close_truncated_block(text: str) -> list[str]:
    """Close-tag completions to try when a tool block was cut off mid-stream.

    ``max_tokens`` routinely truncates the model mid-block. Each completion
    supplies the close tags the block still owes, innermost first, so a partial
    block becomes a usable tool call instead of a silent empty turn.
    """
    stripped = text.rstrip()
    # Only the final (incomplete) block matters; anything before it already
    # parsed or was reported.
    start = max(stripped.rfind("<|DSML|tool_calls"), stripped.rfind("<|dsml|tool_calls"))
    if start != -1:
        stripped = stripped[start:]
    openings = re.findall(r"<\|DSML\|(tool_calls|invoke|parameter)\b[^>]*>", stripped, re.IGNORECASE)
    if not openings:
        return []

    closers = [f"</|DSML|{name}>" for name in reversed(openings)]
    return ["".join(closers[offset:]) for offset in range(len(closers))]


def _recover_trailing_block(
    remainder: str,
    allowed_tool_names: set[str] | None,
    required_params: dict[str, list[str]] | None = None,
) -> tuple[str, list[dict[str, object]]]:
    """Salvage a truncated tool block that would otherwise be discarded."""
    if not remainder or "<|" not in remainder:
        return "", []
    for completion in _close_truncated_block(remainder):
        candidate = remainder.rstrip() + completion
        spans, tool_calls = _extract_tool_blocks(
            candidate, allowed_tool_names, allow_trailing_close=True, required_params=required_params
        )
        if tool_calls:
            leftover = _remove_spans(candidate, spans, trim_outer_whitespace=True)
            return leftover, tool_calls
    return "", []


# A split tag can only ever be this long; scanning the whole buffer to find
# one made every streamed event proportional to the total output so far.
MAX_TAG_HINT_LENGTH = max(len(hint) for hint in TAG_NAME_HINTS)


# A close tag split across two scans can be at most this long.
MAX_CLOSE_TAG_LENGTH = max(
    len("</|dsml|tool_calls>"), len("</|dsmltoolcalls|>"), len("<|/dsmltoolcalls|>")
) + 8

DSML_TOOL_CALLS_OPEN_PREFIX = "<|dsml|tool_calls"


def _is_dsml_open(text: str, index: int | None) -> bool:
    """True when the held region is an unterminated DSML tool_calls block."""
    if index is None or index > 0:
        return False
    return text[: len(DSML_TOOL_CALLS_OPEN_PREFIX)].lower().startswith(DSML_TOOL_CALLS_OPEN_PREFIX)


def _find_partial_tag_start(text: str) -> int | None:
    if not text:
        return None
    window_start = max(0, len(text) - MAX_TAG_HINT_LENGTH)
    lowered_tail = text[window_start:].lower()
    pipe_tag_start = lowered_tail.rfind("<|")
    if pipe_tag_start != -1 and ">" not in lowered_tail[pipe_tag_start:]:
        return window_start + pipe_tag_start
    for hint in TAG_NAME_HINTS:
        lowered_hint = hint.lower()
        max_overlap = min(len(hint), len(lowered_tail))
        for size in range(max_overlap, 0, -1):
            if lowered_tail.endswith(lowered_hint[:size]):
                return window_start + len(lowered_tail) - size
    return None


def _looks_like_tool_markup_fragment(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if lowered.startswith("<|dsml|") or lowered.startswith("</|dsml|") or lowered.startswith("<|/dsml"):
        return True
    if stripped.startswith("<ml_") or stripped.startswith("</ml_"):
        return True
    if stripped.startswith("<tool_") or stripped.startswith("</tool_"):
        return True
    if stripped.startswith("<invoke") or stripped.startswith("</invoke"):
        return True
    if stripped.startswith("<parameter") or stripped.startswith("</parameter"):
        return True
    if stripped.startswith("<m") and any(token in stripped for token in ("ml_", "tool_", "tool_calls", "tool_result")):
        return True
    return False


def _split_stream_text(
    text: str,
    allowed_tool_names: set[str] | None,
    final: bool,
    required_params: dict[str, list[str]] | None = None,
) -> tuple[str, str, list[dict[str, object]], int | None]:
    hold_from_candidates = [
        index
        for index in (_find_unmatched_fence_start(text), _find_incomplete_block_start(text, allow_trailing_close=final))
        if index is not None
    ]

    if not final:
        partial_start = _find_partial_tag_start(text)
        if partial_start is not None:
            hold_from_candidates.append(partial_start)

    if hold_from_candidates:
        safe_end = min(hold_from_candidates)
    else:
        safe_end = len(text)

    processable = text[:safe_end]
    remainder = text[safe_end:]
    spans, tool_calls = _extract_tool_blocks(
        processable, allowed_tool_names, allow_trailing_close=final, required_params=required_params
    )
    visible = _remove_spans(processable, spans, trim_outer_whitespace=final)
    hold_start = safe_end if hold_from_candidates else None
    return visible, remainder, tool_calls, hold_start


def parse_tool_calls_from_text(
    text: str,
    allowed_tool_names: set[str] | None = None,
    required_params: dict[str, list[str]] | None = None,
) -> tuple[str, list[dict[str, object]]]:
    if not text:
        return "", []
    spans, tool_calls = _extract_tool_blocks(
        text, allowed_tool_names, allow_trailing_close=True, required_params=required_params
    )
    if tool_calls:
        return _remove_spans(text, spans), tool_calls

    # No DSML call. Fall back to JSON drift, and either way keep protocol
    # markup out of the text the client shows the user.
    json_spans, json_calls = _extract_json_tool_calls(text, allowed_tool_names, required_params)
    if json_calls:
        return _remove_spans(text, json_spans), json_calls
    return _remove_spans(text, spans), tool_calls


@dataclass
class StreamingToolParser:
    pending_text: str = ""
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    allowed_tool_names: set[str] | None = None
    required_params: dict[str, list[str]] | None = None
    _open_block_start: int | None = None
    _scanned_to: int = 0
    _close_scan_tail: str = ""
    _buffer_has_fence: bool = False
    _buffered_chunks: list[str] = field(default_factory=list)

    def consume(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._buffered_block_is_open(chunk):
            return ""
        self.pending_text += chunk
        visible, self.pending_text, parsed_calls, hold_start = _split_stream_text(
            self.pending_text,
            allowed_tool_names=self.allowed_tool_names,
            final=False,
            required_params=self.required_params,
        )
        self.tool_calls.extend(parsed_calls)
        self._open_block_start = hold_start if _is_dsml_open(self.pending_text, hold_start) else None
        self._scanned_to = len(self.pending_text)
        self._close_scan_tail = self.pending_text[-MAX_CLOSE_TAG_LENGTH :]
        self._buffer_has_fence = "```" in self.pending_text
        return visible

    def _buffered_block_is_open(self, chunk: str) -> bool:
        """Cheaply accumulate while still inside the last open DSML block.

        Replaying the whole argument on every event kept the per-event cost
        proportional to everything generated so far, so a large Write spent
        seconds of parser CPU. The buffer is append-only until the block's
        close tag appears, and the close-tag search only ever re-reads one
        short overlap, making each event O(chunk). Anything unusual — a code
        fence, or a hold that is not a DSML block — falls back to the full
        scan rather than trying to be clever.
        """
        if self._open_block_start is None:
            return False
        if "```" in chunk:
            self._buffer_has_fence = True
        if self._buffer_has_fence:
            self._flush_buffered_block()
            return False

        probe = self._close_scan_tail + chunk
        if DSML_TOOL_CALLS_CLOSE_PATTERN.search(probe) is not None:
            # The block ends here: hand the buffer back so the caller's full
            # scan sees it together with this chunk. The chunk must not also
            # be buffered, or it would be appended twice.
            self._flush_buffered_block()
            return False

        self._buffered_chunks.append(chunk)
        self._close_scan_tail = probe[-MAX_CLOSE_TAG_LENGTH :]
        return True

    def _flush_buffered_block(self) -> None:
        if self._buffered_chunks:
            self.pending_text += "".join(self._buffered_chunks)
            self._buffered_chunks = []
        self._open_block_start = None
        self._close_scan_tail = ""
        self._scanned_to = 0

    def flush(self) -> tuple[str, list[dict[str, object]]]:
        self._flush_buffered_block()
        visible, remainder, parsed_calls, _ = _split_stream_text(
            self.pending_text,
            allowed_tool_names=self.allowed_tool_names,
            final=True,
            required_params=self.required_params,
        )
        self.pending_text = ""
        self.tool_calls.extend(parsed_calls)

        # A block cut off at max_tokens is recovered rather than dropped, so
        # the client never receives a turn that is both text-free and
        # tool-free (which stalls an agent loop).
        if not parsed_calls and remainder:
            recovered_text, recovered_calls = _recover_trailing_block(
                remainder, self.allowed_tool_names, self.required_params
            )
            if recovered_calls:
                self.tool_calls.extend(recovered_calls)
                return (visible + recovered_text).strip(), self.tool_calls
            drifted_calls = parse_json_tool_calls_from_text(
                remainder, self.allowed_tool_names, self.required_params
            )
            if drifted_calls:
                self.tool_calls.extend(drifted_calls)
                return visible.strip(), self.tool_calls

        tail = "" if _looks_like_tool_markup_fragment(remainder) else remainder
        return (visible + tail).strip(), self.tool_calls


JSON_NAME_KEYS = ("name", "tool", "tool_name", "toolName")
JSON_ARGUMENT_KEYS = ("arguments", "params", "parameters", "input", "args")
JSON_CALL_PATTERNS = (
    re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE),
    re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE),
    re.compile(r"(?:^|[^A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\{[\s\S]*?\})\s*\)"),
)


def _json_tool_call_from_object(
    payload: object,
    allowed_tool_names: set[str] | None,
    index: int,
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None

    name = ""
    arguments: object = None
    function = payload.get("function")
    if isinstance(function, dict):
        name = str(function.get("name", "")).strip()
        arguments = function.get("arguments", function.get("parameters"))

    if not name:
        for key in JSON_NAME_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                name = candidate.strip()
                break
    if not name or not _is_allowed_tool_name(name, allowed_tool_names):
        return None

    if arguments is None:
        for key in JSON_ARGUMENT_KEYS:
            if key in payload:
                arguments = payload[key]
                break
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return _build_tool_call(name, arguments, index)


def _extract_json_tool_calls(
    text: str,
    allowed_tool_names: set[str] | None = None,
    required_params: dict[str, list[str]] | None = None,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    """Recover tool calls the model wrote as JSON instead of DSML.

    Hand-prompted XML protocols drift: models emit fenced JSON, a
    ``<tool_call>`` wrapper, or a bare ``Name({...})`` call instead. Without
    this the drift arrives at the client as ordinary assistant prose. The
    matched spans are returned too, so the recovered call is not also emitted
    as text.
    """
    if not text or "{" not in text:
        return [], []

    spans: list[tuple[int, int]] = []
    tool_calls: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for pattern in JSON_CALL_PATTERNS:
        for match in pattern.finditer(text):
            inline_name = match.group(1) if pattern.groups == 2 else ""
            raw_json = match.group(2) if pattern.groups == 2 else match.group(1)
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            if inline_name:
                # ``Name({...})`` — the object is the argument list, not a
                # wrapper that carries its own name/arguments pair.
                body = payload if isinstance(payload, dict) else {}
                looks_like_call = any(key in body for key in JSON_ARGUMENT_KEYS) and any(
                    key in body for key in JSON_NAME_KEYS
                )
                payload = body if looks_like_call else {"name": inline_name, "arguments": body}
            tool_call = _json_tool_call_from_object(payload, allowed_tool_names, len(tool_calls))
            if tool_call is None or _is_empty_args_call(tool_call, required_params):
                continue
            key = (
                str(tool_call["function"]["name"]),
                str(tool_call["function"]["arguments"]),
            )
            if key in seen:
                continue
            seen.add(key)
            tool_calls.append(tool_call)
            spans.append(match.span())

    for offset, tool_call in enumerate(tool_calls):
        tool_call["index"] = offset
    return spans, tool_calls


def parse_json_tool_calls_from_text(
    text: str,
    allowed_tool_names: set[str] | None = None,
    required_params: dict[str, list[str]] | None = None,
) -> list[dict[str, object]]:
    return _extract_json_tool_calls(text, allowed_tool_names, required_params)[1]
