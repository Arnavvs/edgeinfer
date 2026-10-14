"""Hybrid retrieval over the analytics database.

  * dense: multilingual-e5-small (ONNX, CPU) embeddings in a FAISS inner-product index
  * sparse: SQLite FTS5 BM25 over the same chunks
  * fusion: reciprocal rank fusion (k=60), optional source filter
  * every hit carries its source table and row reference, so answers can cite it

Chunks are made at build time (build_index.py): short texts (bios, listings)
are one chunk; long ones (transcripts, captions) are split on sentence
boundaries into ~600-character windows with overlap.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_dir: str, threads: int = 2, max_len: int = 256):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = Path(model_dir).expanduser()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 1, pad_token="<pad>")
        self.dim = None

    def encode(self, texts: list[str], kind: str = "passage", batch: int = 32) -> np.ndarray:
        """e5 convention: 'query: ' / 'passage: ' prefixes; mean pooling; unit norm."""
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch([f"{kind}: {t}" for t in texts[i:i + batch]])
            ids = np.asarray([e.ids for e in enc], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feeds)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        v = np.concatenate(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)
        self.dim = v.shape[1]
        return v


_FTS_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


def fts_query(q: str) -> str:
    words = [w for w in _FTS_WORD.findall(q.lower()) if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words[:24])


class LazyEmbedder:
    """Loads the ONNX model on first use and drops it after `idle_s` without
    queries (~470 MB of RAM on a box that also runs other jobs)."""

    def __init__(self, model_dir: str, threads: int = 2, idle_s: float = 900):
        import threading
        self.args, self.idle = (model_dir, threads), idle_s
        self.emb: Embedder | None = None
        self.last = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True).start()

    def encode(self, texts, kind="passage", batch=32):
        import time
        with self.lock:
            if self.emb is None:
                self.emb = Embedder(*self.args)
            self.last = time.time()
            return self.emb.encode(texts, kind, batch)

    def _reaper(self):
        import gc
        import time
        while True:
            time.sleep(60)
            with self.lock:
                if self.emb is not None and time.time() - self.last > self.idle:
                    self.emb = None
                    gc.collect()


class Retriever:
    def __init__(self, db_path: str, vec_path: str, embedder):
        import faiss
        self.db_path = str(Path(db_path).expanduser())
        self.emb = embedder
        vp = Path(vec_path).expanduser()
        self.index = faiss.read_index(str(vp)) if vp.exists() else None
        self.vec_path, self.vec_mtime = vp, vp.stat().st_mtime if vp.exists() else 0

    def maybe_reload(self) -> None:
        """Pick up a nightly rebuild without restarting the app."""
        import faiss
        if self.vec_path.exists() and self.vec_path.stat().st_mtime != self.vec_mtime:
            self.index = faiss.read_index(str(self.vec_path))
            self.vec_mtime = self.vec_path.stat().st_mtime

    def _con(self) -> sqlite3.Connection:
        c = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def sources(self) -> list[tuple[str, int]]:
        with self._con() as c:
            return [(r[0], r[1]) for r in c.execute("SELECT source, COUNT(*) FROM docs GROUP BY source ORDER BY 2 DESC")]

    def search(self, query: str, k: int = 8, source: str | None = None, pool: int = 60) -> list[dict]:
        self.maybe_reload()
        ranks: dict[int, float] = {}
        with self._con() as c:
            fq = fts_query(query)
            if fq:
                sql = "SELECT rowid FROM docs_fts WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?"
                for r, (rid,) in enumerate(c.execute(sql, (fq, pool))):
                    ranks[rid] = ranks.get(rid, 0) + 1 / (60 + r)
            if self.index is not None and self.emb is not None:
                qv = self.emb.encode([query], kind="query")
                _, ids = self.index.search(qv, pool)
                for r, rid in enumerate(ids[0]):
                    if rid >= 0:
                        ranks[int(rid) + 1] = ranks.get(int(rid) + 1, 0) + 1 / (60 + r)  # docs rowid = vector id + 1
            if not ranks:
                return []
            order = sorted(ranks, key=lambda x: -ranks[x])
            out = []
            for rid in order:
                row = c.execute("SELECT rowid, source, ref, title, text FROM docs WHERE rowid=?", (rid,)).fetchone()
                if row is None or (source and row["source"] != source):
                    continue
                out.append({"id": row["rowid"], "source": row["source"], "ref": row["ref"], "title": row["title"],
                            "text": row["text"], "score": round(ranks[rid], 5)})
                if len(out) >= k:
                    break
            return out
