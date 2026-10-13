"""Build the analytics database + search indexes from a dataset config.

    python -m apps.analyst.build_index CONFIG.json

1. attach the source databases read-only and run the config's build SQL,
   which creates curated tables (clean names, joined, typed) in a fresh file
2. turn the configured text columns into chunks in `docs`
3. index the chunks twice: SQLite FTS5 (BM25) and FAISS (e5 embeddings);
   embeddings are cached by text hash, so a nightly rebuild only embeds new text
4. swap the new files in atomically, so the running app never sees half a build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from .retrieval import Embedder


def chunks(text: str, size: int = 600, overlap: int = 120) -> list[str]:
    text = re.sub(r"[ \t]+", " ", (text or "").strip())
    if len(text) <= size + overlap:
        return [text] if text else []
    parts = re.split(r"(?<=[.!?।\n])\s+", text)
    out, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:] + " " + p
        else:
            cur = (cur + " " + p) if cur else p
    if cur.strip():
        out.append(cur.strip())
    return out


def load_config(path: str) -> tuple[dict, Path]:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text()), p.parent


def build(cfg_path: str, embed_threads: int = 4) -> dict:
    cfg, base = load_config(cfg_path)
    out_db = Path(cfg["analytics_db"]).expanduser()
    out_vec = Path(cfg["vectors"]).expanduser()
    out_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = out_db.with_suffix(".building")
    tmp_db.unlink(missing_ok=True)
    t0 = time.time()
    con = sqlite3.connect(tmp_db, uri=True)
    for name, path in cfg.get("attach", {}).items():
        con.execute(f"ATTACH DATABASE ? AS {name}", (f"file:{Path(path).expanduser()}?mode=ro",))
    con.executescript((base / cfg["build_sql"]).read_text())
    con.commit()
    for name in cfg.get("attach", {}):
        con.execute(f"DETACH DATABASE {name}")

    con.executescript("""
        CREATE TABLE docs(source TEXT, ref TEXT, title TEXT, text TEXT);
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
    """)
    n_docs = {}
    for spec in cfg.get("documents", []):
        n = 0
        for row in con.execute(spec["sql"]).fetchall():
            ref, title, text = (str(x) if x is not None else "" for x in row[:3])
            for ch in chunks(text, spec.get("chunk", 600)):
                con.execute("INSERT INTO docs(source, ref, title, text) VALUES (?,?,?,?)", (spec["source"], ref, title, ch))
                n += 1
        n_docs[spec["source"]] = n
    con.executescript("""
        CREATE VIRTUAL TABLE docs_fts USING fts5(title, text, content='docs', content_rowid='rowid',
                                                 tokenize='unicode61 remove_diacritics 2');
        INSERT INTO docs_fts(docs_fts) VALUES('rebuild');
    """)

    # embeddings, cached by content hash across builds
    rows = con.execute("SELECT rowid, title, text FROM docs ORDER BY rowid").fetchall()
    cache_path = out_db.with_name("embed_cache.db")
    cache = sqlite3.connect(cache_path)
    cache.execute("CREATE TABLE IF NOT EXISTS emb(h TEXT PRIMARY KEY, v BLOB)")
    texts = [f"{t}: {x}" if t else x for _, t, x in rows]
    hashes = [hashlib.sha1(s.encode()).hexdigest() for s in texts]
    have = {}
    for i in range(0, len(hashes), 500):
        part = hashes[i:i + 500]
        q = f"SELECT h, v FROM emb WHERE h IN ({','.join('?' * len(part))})"
        have.update({h: np.frombuffer(v, dtype=np.float32) for h, v in cache.execute(q, part)})
    todo = [i for i, h in enumerate(hashes) if h not in have]
    t1 = time.time()
    if todo:
        emb = Embedder(cfg["embedder"], threads=embed_threads)
        B = 256
        for s in range(0, len(todo), B):
            idx = todo[s:s + B]
            vecs = emb.encode([texts[i] for i in idx], kind="passage")
            cache.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                              [(hashes[i], v.astype(np.float32).tobytes()) for i, v in zip(idx, vecs)])
            cache.commit()
            for i, v in zip(idx, vecs):
                have[hashes[i]] = v
            print(f"\rembedded {min(s + B, len(todo))}/{len(todo)} new chunks", end="", flush=True)
        print()
    embed_s = time.time() - t1
    import faiss
    dim = len(next(iter(have.values()))) if have else 384
    index = faiss.IndexFlatIP(dim)
    if rows:
        index.add(np.stack([have[h] for h in hashes]).astype(np.float32))
    tmp_vec = out_vec.with_suffix(".building")
    faiss.write_index(index, str(tmp_vec))

    tables = {}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'docs%' "
                            "AND name NOT IN ('meta') ORDER BY name"):
        tables[t] = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    info = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "tables": tables, "docs": n_docs,
            "chunks": len(rows), "embedded_new": len(todo), "embed_s": round(embed_s, 1),
            "build_s": round(time.time() - t0, 1)}
    con.execute("INSERT INTO meta VALUES ('build', ?)", (json.dumps(info),))
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp_db.replace(out_db)
    tmp_vec.replace(out_vec)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(build(a.config, a.threads), indent=1))


if __name__ == "__main__":
    main()
