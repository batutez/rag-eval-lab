import re
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parents[1] / "data" / "text" / "doc01_yapay_zeka_eylem_plani.md"

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
BLOCK = re.compile(r"[^\n]*\S[^\n]*(?:\n[^\n]*\S[^\n]*)*")   # consecutive non-blank lines
PAGE_MARKER = re.compile(r"<!-- page: pdf (\d+) \| printed (\S+) -->")
HEADING = re.compile(r"(#+)\s+(.*)")
EYLEM = re.compile(r"Eylem (\d+)\s*-")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9“\"'(])")   # end of sentence, next word capitalised


def parse_paragraphs(path, min_chars=40):
    text = path.read_text(encoding="utf-8")
    front = FRONT_MATTER.match(text)
    doc_id = re.search(r"doc_id:\s*(\S+)", front.group(1)).group(1)
    page_pdf = page_printed = None
    headings = {}                                   # heading level -> heading text
    paragraphs = []

    for m in BLOCK.finditer(text, front.end()):
        block = m.group().strip()
        page = PAGE_MARKER.fullmatch(block)
        heading = HEADING.fullmatch(block)
        if page:
            page_pdf, page_printed = int(page.group(1)), page.group(2)
        elif block.startswith("<!--"):
            continue
        elif heading:
            level = len(heading.group(1))
            headings = {l: h for l, h in headings.items() if l < level}
            headings[level] = heading.group(2)
        else:
            if len(block) < min_chars:              # drop labels, sidebar numbers, fragments
                continue
            eylem = EYLEM.match(headings.get(4, ""))
            paragraphs.append({
                "text": " ".join(block.split()),
                "doc_id": doc_id,
                "page_pdf": page_pdf,
                "page_printed": page_printed,
                "heading_path": list(headings.values()),
                "eylem": int(eylem.group(1)) if eylem else None,
                "char_start": m.start(),
                "char_end": m.end(),
            })
    return paragraphs


def split_sentences(text):
    return SENTENCE_END.split(text)


def split_long_sentence(sentence, max_chars):
    """Fallback: cut a sentence longer than max_chars at the last space before the limit."""
    pieces = []
    while len(sentence) > max_chars:
        cut = sentence.rfind(" ", 0, max_chars + 1)
        if cut <= 0:                                # no space at all: hard cut
            cut = max_chars
        pieces.append(sentence[:cut])
        sentence = sentence[cut:].lstrip()
    pieces.append(sentence)
    return pieces


def pack_paragraph(text, max_chars=1000, overlap=True):
    """Fill chunks with whole sentences up to max_chars; repeat the last sentence in the next chunk if it fits."""
    units = [piece for s in split_sentences(text) for piece in split_long_sentence(s, max_chars)]
    chunks, current = [], []
    for unit in units:
        if current and len(" ".join(current + [unit])) > max_chars:
            chunks.append(" ".join(current))
            if overlap and len(current[-1]) + 1 + len(unit) <= max_chars:
                current = [current[-1], unit]
            else:
                current = [unit]
        else:
            current.append(unit)
    chunks.append(" ".join(current))
    return chunks


def chunk_paragraphs(paragraphs, max_chars=1000, overlap=True):
    """One record per piece; every piece keeps its paragraph's metadata (including the paragraph's char span).
    embed_text = heading path + piece, so the embedding sees where the piece belongs."""
    chunks = []
    for paragraph in paragraphs:
        pieces = pack_paragraph(paragraph["text"], max_chars, overlap)
        for i, piece in enumerate(pieces, start=1):
            chunks.append({
                **paragraph,
                "chunk_id": f"{paragraph['doc_id']}-{len(chunks) + 1:04d}",
                "text": piece,
                "embed_text": " > ".join(paragraph["heading_path"]) + "\n\n" + piece,
                "part": i,
                "n_parts": len(pieces),
            })
    return chunks


if __name__ == "__main__":
    import hashlib
    import json
    import statistics

    import wandb
    from transformers import AutoTokenizer

    MAX_CHARS, OVERLAP, MIN_PARAGRAPH_CHARS = 1000, True, 40
    MODEL_ID, MAX_TOKENS, PREFIX = "intfloat/multilingual-e5-large", 512, "passage: "
    OUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / f"{DOC_PATH.stem}_chunks_max{MAX_CHARS}_ov1.jsonl"

    paragraphs = parse_paragraphs(DOC_PATH, MIN_PARAGRAPH_CHARS)
    chunks = chunk_paragraphs(paragraphs, MAX_CHARS, OVERLAP)

    # step 6: token check with the embedding tokenizer (prefix and special tokens included)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    for c in chunks:
        c["n_tokens"] = len(tokenizer(PREFIX + c["embed_text"])["input_ids"])
    over_limit = [c["chunk_id"] for c in chunks if c["n_tokens"] > MAX_TOKENS]

    # step 7: save
    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    chars = [len(c["text"]) for c in chunks]
    tokens = [c["n_tokens"] for c in chunks]
    stats = {
        "n_paragraphs": len(paragraphs),
        "n_chunks": len(chunks),
        "n_split_paragraphs": sum(c["part"] == 1 and c["n_parts"] > 1 for c in chunks),
        "chars_min": min(chars), "chars_median": statistics.median(chars), "chars_max": max(chars),
        "tokens_min": min(tokens), "tokens_median": statistics.median(tokens), "tokens_max": max(tokens),
        "n_chunks_over_token_limit": len(over_limit),
    }
    print(json.dumps(stats, indent=2))
    print("over token limit:", over_limit or "none")
    print("saved:", OUT_PATH)

    # step 8: W&B (config, summary, chunks file as a private artifact)
    with wandb.init(project="rag-eval-lab", name=f"m1-{DOC_PATH.stem[:5]}-max{MAX_CHARS}-ov1", group="m1-chunking", config={
        "document": DOC_PATH.name,
        "document_sha256": hashlib.sha256(DOC_PATH.read_bytes()).hexdigest(),
        "language": "tr",
        "min_paragraph_chars": MIN_PARAGRAPH_CHARS,
        "split_method": "sentence packing + fallback cut at space",
        "max_chars": MAX_CHARS,
        "overlap": "1 sentence" if OVERLAP else "none",
        "embed_text": "heading path + newline + piece",
        "token_check_model": MODEL_ID,
        "token_check_prefix": PREFIX,
        "max_tokens": MAX_TOKENS,
    }) as run:
        run.summary.update(stats)
        artifact = wandb.Artifact(name=f"{DOC_PATH.stem[:5]}-chunks", type="dataset")
        artifact.add_file(str(OUT_PATH))
        run.log_artifact(artifact)
