"""Corpus loading and chunking."""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    ordinal: int


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware sliding window: pack paragraphs up to ``size`` chars,
    carrying ``overlap`` characters of tail context into the next chunk."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > size:
            chunks.append(buf.strip())
            buf = (buf[-overlap:] + "\n\n" + para) if overlap else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def load_corpus(corpus_dir: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        source = os.path.splitext(os.path.basename(path))[0]
        for i, piece in enumerate(_split(body, size, overlap)):
            chunks.append(Chunk(text=piece, source=source, ordinal=i))
    return chunks


def corpus_fingerprint(corpus_dir: str) -> str:
    """A hash of every corpus file's name + content, so the index rebuilds
    automatically whenever the corpus changes (e.g. across a redeploy)."""
    h = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()
