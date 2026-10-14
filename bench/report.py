"""Results -> markdown tables + SVG charts for the README.

    python bench/report.py RESULTS_DIR docs/results.md docs/img/
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ENGINE_ORDER = ["edgeinfer", "llama.cpp", "llama-cpp-python", "ollama", "transformers"]
COLORS = {"edgeinfer": "#2a6fd6", "llama.cpp": "#e07b39", "llama-cpp-python": "#d4a72c", "ollama": "#1a8a5f",
          "transformers": "#8c6bc8", "llama-server": "#e07b39"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bar_svg(title: str, groups: list[str], series: dict[str, list[float | None]], unit: str, path: Path) -> None:
    """Grouped horizontal bars; colours per series; works in light and dark (no background)."""
    names = list(series)
    bh, gap, left, width = 16, 10, 150, 640
    vmax = max([v for vs in series.values() for v in vs if v] + [1])
    h = 40 + len(groups) * (len(names) * bh + gap) + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {h}" font-family="system-ui,sans-serif" font-size="12">',
           '<style>text{fill:#6f6e68}.t{font-size:14px;font-weight:600;fill:#888}</style>',
           f'<text class="t" x="0" y="18">{title}</text>']
    y = 36
    for gi, g in enumerate(groups):
        out.append(f'<text x="0" y="{y + len(names) * bh / 2 + 4}">{g}</text>')
        for n in names:
            v = series[n][gi]
            if v:
                w = (width - left - 90) * v / vmax
                out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bh - 3}" rx="3" fill="{COLORS.get(n, "#999")}"/>')
                out.append(f'<text x="{left + w + 6:.1f}" y="{y + bh - 5}">{v:g} {unit}</text>')
            y += bh
        y += gap
    lx = 0
    for n in names:
        out.append(f'<rect x="{lx}" y="{h - 14}" width="10" height="10" rx="2" fill="{COLORS.get(n, "#999")}"/>'
                   f'<text x="{lx + 14}" y="{h - 5}">{n}</text>')
        lx += 22 + 7 * len(n)
    out.append("</svg>")
    path.write_text("\n".join(out))


def main():
    res, out_md, img = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    img.mkdir(parents=True, exist_ok=True)
    md = []

    raw = [r for r in load(res / "raw.jsonl") if "tok_s" in r]
    if raw:
        latest = {}
        for r in raw:  # last measurement wins
            latest[(r["model"], r["engine"], r["test"])] = r
        models = sorted({k[0] for k in latest}, key=lambda m: float(m.split("-")[-1].rstrip("bB")) if m[-1] in "bB" else 0)
        engines = [e for e in ENGINE_ORDER if any(k[1] == e for k in latest)] + \
                  sorted({k[1] for k in latest} - set(ENGINE_ORDER))
        for test, label in [("pp512", "Prompt processing, 512 tokens (tok/s)"), ("tg128", "Generation, 128 tokens (tok/s)"),
                            ("tg64@d4096", "Generation with 4K context (tok/s)")]:
            md.append(f"\n#### {label}\n\n| model | " + " | ".join(engines) + " |\n|---|" + "---|" * len(engines))
            for m in models:
                cells = []
                for e in engines:
                    r = latest.get((m, e, test))
                    cells.append(f"{r['tok_s']:.1f}" if r else "–")
                md.append(f"| {m} | " + " | ".join(cells) + " |")
            bar_svg(label, models, {e: [latest.get((m, e, test), {}).get("tok_s") for m in models] for e in engines},
                    "tok/s", img / f"{test.replace('@', '_')}.svg")
        md.append("\n| model | engine | file MB | peak RSS MB |\n|---|---|---|---|")
        for m in models:
            for e in engines:
                r = latest.get((m, e, "pp512"))
                if r:
                    md.append(f"| {m} | {e} | {r.get('file_mb') or '–'} | {r.get('peak_rss_mb') or '–'} |")

    for f in sorted(res.glob("quality-*.jsonl")):
        rows = load(f)
        if not rows:
            continue
        best = {}
        for r in rows:
            best[r["model"]] = r
        md.append(f"\n#### Quality, {f.stem.split('-', 1)[1]} (wikitext-2, vs the higher-precision base)\n\n"
                  "| model file | KL mean | KL p99 | top-1 agree | perplexity |\n|---|---|---|---|---|")
        for r in best.values():
            md.append(f"| {r['model']} | {r['kld_mean']:.4f} | {r['kld_p99']:.3f} | {100 * r['top1_agree']:.1f}% | {r['ppl']:.3f} |")

    serve = [r for r in load(res / "serve.jsonl") if "scenario" in r]
    if serve:
        by = defaultdict(list)
        for r in serve:
            by[(r["model"], r["server"], r["scenario"])].append(r)
        models = sorted({k[0] for k in by})
        servers = [s for s in ["edgeinfer", "llama-server", "ollama"] if any(k[1] == s for k in by)]
        for m in models:
            md.append(f"\n#### Multi-turn chat, {m}: time to first token per turn (s)\n\n| turn | " +
                      " | ".join(servers) + " |\n|---|" + "---|" * len(servers))
            turns = max((len(by[(m, s, "multiturn")]) for s in servers), default=0)
            for t in range(turns):
                cells = []
                for s in servers:
                    rs = by.get((m, s, "multiturn"), [])
                    cells.append(f"{rs[t]['ttft_s']:.2f}" if t < len(rs) else "–")
                md.append(f"| {t + 1} | " + " | ".join(cells) + " |")
            md.append("\n| server | mean decode tok/s | " + " | ".join(
                f"{c} clients tok/s" for c in sorted({r['clients'] for s in servers for r in by.get((m, s, 'concurrency'), [])})) +
                " |\n|---|---|" + "---|" * len({r['clients'] for s in servers for r in by.get((m, s, 'concurrency'), [])}))
            for s in servers:
                mt = by.get((m, s, "multiturn"), [])
                dec = [r["decode_tok_s"] for r in mt if r.get("decode_tok_s")]
                conc = sorted(by.get((m, s, "concurrency"), []), key=lambda r: r["clients"])
                md.append(f"| {s} | {sum(dec) / len(dec):.1f} | " + " | ".join(f"{r['agg_tok_s']:.1f}" for r in conc) + " |"
                          if dec else f"| {s} | – | |")
            ttft = {s: [r["ttft_s"] for r in by.get((m, s, "multiturn"), [])] for s in servers}
            bar_svg(f"{m}: time to first token per chat turn (lower is better)",
                    [f"turn {i + 1}" for i in range(max(len(v) for v in ttft.values()))],
                    {s: (v + [None] * 10)[:max(len(x) for x in ttft.values())] for s, v in ttft.items()}, "s",
                    img / f"ttft_{m}.svg")

    agent = [r for r in load(res / "analyst.jsonl") if "accuracy" in r]
    if agent:
        md.append("\n#### Data Desk agent (SQL + search tools)\n\n| model | accuracy | tool errors | model calls/question |"
                  " mean s | prompt tokens from cache |\n|---|---|---|---|---|---|")
        for r in agent:
            cache = r["cached_tokens"] / r["prompt_tokens"] if r["prompt_tokens"] else 0
            md.append(f"| {r['model']} | {100 * r['accuracy']:.0f}% ({r['questions']} q) | {100 * r['tool_error_rate']:.0f}% | "
                      f"{r['mean_model_calls']} | {r['mean_s']} | {100 * cache:.0f}% |")
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md} and charts in {img}")


if __name__ == "__main__":
    main()
