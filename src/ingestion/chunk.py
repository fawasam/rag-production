"""Phase 1 chunking: fixed-size token windows with overlap.

Naive but deterministic — good enough for a baseline. Swap for
structure-aware chunking (headings, sentences) in Phase 2 if needed.
"""
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from src.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_file

ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    position: int


def load_documents(raw_dir: Path) -> list[tuple[str, Path, str]]:
    """Returns list of (doc_id, source_path, text) for every supported file in raw_dir.

    Supported formats: .md, .txt, .pdf, .docx (see parsers.SUPPORTED_EXTENSIONS).
    doc_id uses the filename including extension to avoid collisions between,
    e.g., report.pdf and report.docx.
    """
    docs = []
    for path in sorted(raw_dir.glob("**/*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            try:
                text = parse_file(path)
            except Exception as exc:
                print(f"Skipping {path} — failed to parse: {exc}")
                continue
            if not text.strip():
                print(f"Skipping {path} — no extractable text (scanned/image PDF?)")
                continue
            docs.append((path.name, path, text))
    return docs


def chunk_text(doc_id: str, source_path: str, text: str) -> list[Chunk]:
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    position = 0
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    while start < len(tokens):
        window = tokens[start : start + CHUNK_SIZE_TOKENS]
        chunk_text_str = ENCODING.decode(window)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk_{position}",
                doc_id=doc_id,
                source_path=source_path,
                text=chunk_text_str.strip(),
                position=position,
            )
        )
        position += 1
        start += step
    return chunks


def chunk_all(raw_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc_id, path, text in load_documents(raw_dir):
        all_chunks.extend(chunk_text(doc_id, str(path), text))
    return all_chunks


if __name__ == "__main__":
    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    chunks = chunk_all(raw)
    print(f"Loaded {len(chunks)} chunks from {raw}")
    for c in chunks[:3]:
        print(f"--- {c.chunk_id} ---\n{c.text[:200]}...\n")
