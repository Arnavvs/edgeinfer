"""The Python chat renderer must reproduce Qwen3.5's Jinja chat template.

    python tests/test_chat.py HF_DIR
"""
import json
import sys
from pathlib import Path

import jinja2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeinfer.chat import ChatTokenizer, cache_marks, parse_tool_calls, render  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "sql", "description": "Run a read-only SQL query",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

CASES = [
    ([{"role": "user", "content": "hi"}], None),
    ([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"}], None),
    ([{"role": "system", "content": "S"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
      {"role": "user", "content": "c"}], None),
    ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nb"},
      {"role": "user", "content": "c"}], None),
    ([{"role": "system", "content": "Use tools."}, {"role": "user", "content": "how many creators?"},
      {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
          "name": "sql", "arguments": {"query": "SELECT COUNT(*) FROM creators"}}}]},
      {"role": "tool", "content": "[[117]]"}], TOOLS),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "Let me check.", "tool_calls": [
        {"type": "function", "function": {"name": "sql", "arguments": "{\"query\": \"SELECT 1\"}"}}]},
      {"role": "tool", "content": "1"}, {"role": "tool", "content": "2"}, {"role": "assistant", "content": "done"},
      {"role": "user", "content": "thanks"}], TOOLS),
]


def main():
    hf = Path(sys.argv[1])
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)

    def raise_exception(msg):
        raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    tmpl = env.from_string((hf / "chat_template.jinja").read_text())
    ok = True
    for i, (msgs, tools) in enumerate(CASES):
        for thinking in (False, True):
            ref_msgs = json.loads(json.dumps(msgs))
            for m in ref_msgs:  # the Jinja template expects mapping arguments
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc["function"]["arguments"], str):
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            ref = tmpl.render(messages=ref_msgs, tools=tools, add_generation_prompt=True, enable_thinking=thinking)
            got = render(msgs, tools, True, thinking)
            same = ref == got
            ok &= same
            if not same:
                import difflib
                print(f"case {i} thinking={thinking} MISMATCH")
                print("".join(difflib.unified_diff(ref.splitlines(True), got.splitlines(True))))
    print(f"template cases: {'all match' if ok else 'FAIL'}")

    # cache marks must be exact token prefixes of the prompt
    tok = ChatTokenizer(hf)
    msgs = CASES[2][0]
    prompt = tok.chat(msgs, keep_think=True)
    marks = cache_marks(tok, msgs, prompt, None, keep_think=True)
    print("cache marks", marks, "of", len(prompt))
    ok &= bool(marks) and all(0 < m < len(prompt) for m in marks)

    content, calls = parse_tool_calls("I'll query.\n\n<tool_call>\n<function=sql>\n<parameter=query>\n"
                                      "SELECT city, COUNT(*) FROM creators GROUP BY city\n</parameter>\n</function>\n</tool_call>")
    ok &= content == "I'll query." and calls and json.loads(calls[0]["function"]["arguments"])["query"].startswith("SELECT")
    print("tool-call parse", content, calls[0]["function"])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
