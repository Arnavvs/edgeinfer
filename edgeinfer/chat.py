"""Tokenizer, chat template and tool-call parsing for Qwen3.5.

The renderer reproduces Qwen3.5's Jinja chat template (tests/test_chat.py
checks it against the real template) with one optional change:
`keep_think=True` keeps the empty "<think>\\n\\n</think>\\n\\n" block of earlier
assistant turns. The model generated those tokens, so keeping them makes the
next turn's prompt an exact extension of the cached sequence and the whole
history is served from the prefix cache instead of being re-encoded.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

TOOLS_PREAMBLE = (
    "# Tools\n\nYou have access to the following functions:\n\n<tools>"
)
TOOLS_RULES = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
    "but NOT after\n- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n</IMPORTANT>"
)


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            out.append(item.get("text", ""))
        elif isinstance(item, dict) and "text" in item:
            out.append(item["text"])
        else:
            raise ValueError("only text content is supported")
    return "".join(out)


def _tojson(x) -> str:
    # matches Jinja's tojson for the tool schema line (compact separators, keep unicode)
    return json.dumps(x, ensure_ascii=False)


def render(messages: list[dict], tools: list | None = None, add_generation_prompt: bool = True,
           enable_thinking: bool = False, keep_think: bool = False, last_query: int | None = None) -> str:
    """Renders like Qwen3.5's chat template. `last_query` overrides the index of
    the last real user message (used to render a prefix of a conversation
    exactly as it appears inside the full conversation)."""
    if not messages:
        raise ValueError("no messages")
    out = []
    first_sys = messages[0].get("role") == "system"
    if tools:
        out.append(f"{IM_START}system\n{TOOLS_PREAMBLE}")
        for t in tools:
            out.append("\n" + _tojson(t))
        out.append(TOOLS_RULES)
        if first_sys:
            c = _text(messages[0]["content"]).strip()
            if c:
                out.append("\n\n" + c)
        out.append(f"{IM_END}\n")
    elif first_sys:
        out.append(f"{IM_START}system\n{_text(messages[0]['content']).strip()}{IM_END}\n")

    if last_query is None:
        last_query = find_last_query(messages)

    for i, m in enumerate(messages):
        role = m["role"]
        content = _text(m.get("content")).strip()
        if role == "system":
            if i != 0:
                raise ValueError("system message must be first")
        elif role == "user":
            out.append(f"{IM_START}user\n{content}{IM_END}\n")
        elif role == "assistant":
            reasoning = m.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = ""
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = content.split("</think>")[-1].lstrip("\n")
            reasoning = reasoning.strip()
            if i > last_query or (keep_think and not reasoning):
                out.append(f"{IM_START}assistant\n<think>\n{reasoning}\n</think>\n\n{content}")
            else:
                out.append(f"{IM_START}assistant\n{content}")
            for j, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function", tc)
                if j == 0:
                    out.append("\n\n<tool_call>\n" if content.strip() else "<tool_call>\n")
                else:
                    out.append("\n<tool_call>\n")
                out.append(f"<function={fn['name']}>\n")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                for k, v in args.items():
                    v = _tojson(v) if isinstance(v, (dict, list)) else str(v)
                    out.append(f"<parameter={k}>\n{v}\n</parameter>\n")
                out.append("</function>\n</tool_call>")
            out.append(f"{IM_END}\n")
        elif role == "tool":
            prev = messages[i - 1]["role"] if i > 0 else None
            if prev != "tool":
                out.append(f"{IM_START}user")
            out.append(f"\n<tool_response>\n{content}\n</tool_response>")
            nxt = messages[i + 1]["role"] if i + 1 < len(messages) else None
            if nxt != "tool":
                out.append(f"{IM_END}\n")
        else:
            raise ValueError(f"unexpected role {role}")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
        out.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(out)


def find_last_query(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] == "user":
            c = _text(m["content"]).strip()
            if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                return i
    raise ValueError("no user query found in messages")


def cache_marks(tok: "ChatTokenizer", messages: list[dict], prompt: list[int], tools=None,
                keep_think: bool = False) -> list[int]:
    """Token offsets worth a prefix-cache snapshot: the end of the system
    prompt (shared by every conversation that uses it) and the start of the
    last user turn (so regenerating or editing the last message is cheap)."""
    lq = find_last_query(messages)
    cuts = set()
    if messages[0]["role"] == "system" or tools:
        cuts.add(1)
    cuts.add(lq)
    marks = []
    for k in sorted(cuts):
        if k <= 0 or k >= len(messages):
            continue
        text = render(messages[:k], tools, add_generation_prompt=False, keep_think=keep_think, last_query=lq)
        ids = tok.encode(text)
        if 0 < len(ids) < len(prompt) and prompt[:len(ids)] == ids:
            marks.append(len(ids))
    return marks


_TC = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Splits model output into (content, OpenAI-style tool_calls)."""
    calls = []
    for m in _TC.finditer(text):
        args = {}
        for p in _PARAM.finditer(m.group(2)):
            v = p.group(2)
            try:
                args[p.group(1)] = json.loads(v)
            except (ValueError, TypeError):
                args[p.group(1)] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
    content = text[:text.find("<tool_call>")] if calls else text
    return content.strip() if calls else content, calls


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, answer) for output generated with thinking enabled."""
    if "</think>" in text:
        r, a = text.split("</think>", 1)
        return r.replace("<think>", "").strip(), a.lstrip("\n")
    return "", text


class ChatTokenizer:
    def __init__(self, path: str | Path):
        p = Path(path)
        self.tok = Tokenizer.from_file(str(p / "tokenizer.json" if p.is_dir() else p))
        self.im_end = self.tok.token_to_id(IM_END)
        self.eos = self.tok.token_to_id("<|endoftext|>")
        self.tool_call = self.tok.token_to_id("<tool_call>")
        self.stop_ids = {t for t in (self.im_end, self.eos) if t is not None}

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special: bool = False) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    def chat(self, messages, tools=None, enable_thinking=False, keep_think=False) -> list[int]:
        return self.encode(render(messages, tools, True, enable_thinking, keep_think))


class StreamDecoder:
    """Incremental detokenizer: emits text only once it is valid UTF-8."""

    def __init__(self, tok: ChatTokenizer):
        self.tok = tok
        self.ids: list[int] = []
        self.prefix = 0  # ids[prefix:read] were already turned into text
        self.read = 0
        self.text = ""

    def push(self, tid: int) -> str:
        self.ids.append(tid)
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        cur = self.tok.decode(self.ids[self.prefix:])
        if cur.endswith("�"):
            return ""
        delta = cur[len(prev):]
        self.prefix = self.read
        self.read = len(self.ids)
        self.text += delta
        return delta
