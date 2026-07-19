"""
Chunk the noon KB PDFs by their horizontal separator lines (the thin gray
rules Zoho draws between sections), instead of by font-size heading
hierarchy.

Why: these rule lines are a much more reliable signal in this template than
font size/bold detection, since they're actual vector drawings in the PDF,
independent of how the text itself is styled.

Logic:
  - Detect horizontal rule lines per page (thin, wide rectangles).
  - Walk all text lines on the page in top-to-bottom order; every time a
    line's y-position passes the next rule line, close the current chunk
    and start a new one.
  - A document with ZERO rule lines never triggers a split, so all its text
    naturally ends up in a single chunk (matches: "if a page doesn't have
    lines, make it one chunk" -- no special-casing needed).
  - Small leftover chunks (like a lone logo/header block before the first
    rule line) are dropped as noise.
  - The article's title (first line of the first real chunk) is prefixed
    onto every other chunk, so a section like "Before Your Order Is Packed"
    doesn't lose the context of which article it belongs to.

Run:
    pip install pymupdf
    python chunk_kb_by_separators.py
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_FOLDER = "/Users/tokamohamed/Downloads/AI-Customer-Support-Agent-for-E-commerce/data"
OUTPUT_PATH = "chunks.json"

MIN_CHUNK_WORDS = 4    # drop tiny leftover chunks (e.g. just a logo/header)
MAX_WORDS = 250        # safety cap applied to separator-based chunks
OVERLAP_WORDS = 40     # only used if a separator-based chunk exceeds MAX_WORDS

# Dedicated fallback for documents with ZERO detected separator lines (some
# PDFs lose their line breaks on export, so there's nothing to split on
# structurally -- these get a plain fixed-size sliding window instead).
NO_SEP_CHUNK_WORDS = 150
NO_SEP_OVERLAP_WORDS = 20


# ---------------------------------------------------------------------------
# Step 1: detect horizontal rule lines on a page
# ---------------------------------------------------------------------------

def get_horizontal_separators(page, min_width_ratio=0.6, max_height=2.0):
    page_width = page.rect.width
    seps = []
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None:
            continue
        if rect.height <= max_height and rect.width >= page_width * min_width_ratio:
            seps.append(rect.y0)
    return sorted(set(round(y, 1) for y in seps))


# ---------------------------------------------------------------------------
# Step 2: extract text lines with their vertical position
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+")


def extract_lines_with_position(page):
    lines = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            # strip URLs -- these are page-footer artifacts (the article's
            # own KB link), not real content. Drop the line entirely if it
            # was JUST a URL; otherwise keep the surrounding text.
            text = URL_PATTERN.sub("", text).strip()
            if not text:
                continue
            y0 = line["bbox"][1]
            lines.append({"y": y0, "text": text})
    return lines


# ---------------------------------------------------------------------------
# Step 3: walk lines in order, splitting at each separator crossing
# ---------------------------------------------------------------------------

def build_raw_sections(pdf_path: Path):
    """
    Returns (sections, has_separators, full_text):
      sections      - list of list-of-lines, split at each separator crossing
      has_separators - True if ANY horizontal rule was found anywhere in the doc
      full_text     - all lines concatenated, in order (used for the
                       no-separator fixed-size fallback)
    """
    doc = fitz.open(pdf_path)
    sections = []
    current = []
    has_separators = False
    all_lines = []

    for page in doc:
        seps = get_horizontal_separators(page)
        lines = extract_lines_with_position(page)
        if seps:
            has_separators = True

        sep_idx = 0
        for line in lines:
            all_lines.append(line["text"])
            while sep_idx < len(seps) and line["y"] > seps[sep_idx]:
                if current:
                    sections.append(current)
                current = []
                sep_idx += 1
            current.append(line["text"])

    if current:
        sections.append(current)

    doc.close()
    full_text = "\n".join(all_lines)
    return sections, has_separators, full_text


# ---------------------------------------------------------------------------
# Step 4: turn raw sections into chunks, drop noise, prefix the doc title
# ---------------------------------------------------------------------------

def filename_to_title(source_name: str) -> str:
    """
    Derive a readable title from the filename, for the no-separator fallback
    path -- these docs have no reliable title line to extract (flat text
    with no structure), so the filename is a safer bet.
    """
    stem = Path(source_name).stem
    # strip trailing date-like suffixes, e.g. "-6-3-2024", "-28-1-2026"
    stem = re.sub(r"(-\d{1,2}-\d{1,2}-\d{2,4})$", "", stem)
    stem = re.sub(r"(-\d{1,2}-\d{4})$", "", stem)
    words = stem.replace("_", "-").split("-")
    return " ".join(w if w.isupper() else w.capitalize() for w in words if w)


def fixed_size_chunks(full_text: str, source_name: str, path: str,
                       chunk_words: int = NO_SEP_CHUNK_WORDS,
                       overlap_words: int = NO_SEP_OVERLAP_WORDS):
    """
    Plain sliding-window chunker for documents with zero detected
    separators -- there's no structural signal to split on, so fall back to
    fixed size + overlap.
    """
    words = full_text.split()
    if not words:
        return []

    title = filename_to_title(source_name)
    chunks = []
    start = 0
    part = 1
    while start < len(words):
        end = start + chunk_words
        piece = " ".join(words[start:end])
        chunks.append({
            "text": f"Title: {title}\n\n{piece}",
            "metadata": {"source": source_name, "path": path, "part": part,
                         "chunking": "fixed_fallback"},
        })
        start += chunk_words - overlap_words
        part += 1
    return chunks


def count_meaningful_sections(sections):
    """
    Filter out tiny leftover sections (e.g. a lone logo/header block before
    the first rule line), the same way sections_to_chunks does. Used to
    decide routing -- a doc with only 1 meaningful section after filtering
    has no real internal structure to split on, even if a boilerplate
    separator (like the one under the header) was technically detected.
    """
    if len(sections) > 1:
        filtered = [s for s in sections if len(" ".join(s).split()) >= MIN_CHUNK_WORDS]
        return filtered or sections
    return sections


def sections_to_chunks(source_name: str, path: str, sections):
    # drop tiny leftover sections (e.g. a lone logo/header block), but never
    # let this empty out a document that only has one section to begin with
    if len(sections) > 1:
        sections = [s for s in sections if len(" ".join(s).split()) >= MIN_CHUNK_WORDS] or sections

    if not sections:
        return []

    doc_title = sections[0][0] if sections[0] else source_name

    chunks = []
    for i, sec_lines in enumerate(sections):
        text = "\n".join(sec_lines)
        chunks.append({
            "text": f"Title: {doc_title}\n\n{text}",
            "metadata": {"source": source_name, "path": path, "part": i + 1},
        })
    return chunks


# ---------------------------------------------------------------------------
# Step 5: safety-net split for any chunk that runs unusually long
# ---------------------------------------------------------------------------

def enforce_max_length(chunks, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS):
    final_chunks = []
    for c in chunks:
        words = c["text"].split()
        if len(words) <= max_words:
            final_chunks.append(c)
            continue

        start = 0
        sub_part = 1
        while start < len(words):
            end = start + max_words
            piece = " ".join(words[start:end])
            final_chunks.append({
                "text": piece,
                "metadata": {**c["metadata"], "sub_part": sub_part},
            })
            start += max_words - overlap_words
            sub_part += 1
    return final_chunks


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

def main():
    directory = Path(TARGET_FOLDER)
    pdf_files = sorted(directory.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    all_chunks = []
    for pdf_path in pdf_files:
        try:
            sections, _has_separators, full_text = build_raw_sections(pdf_path)
            meaningful = count_meaningful_sections(sections)
            word_count = len(full_text.split())
            # Use the dedicated fixed-size fallback ONLY for genuinely long,
            # structureless docs (no real sections AND long enough that a
            # single chunk would be unwieldy). Short single-section docs
            # still go through sections_to_chunks -- it produces one clean
            # chunk with a proper title and no header-noise, which is
            # exactly right for a normal one-page Q&A article.
            if len(meaningful) > 1 or word_count <= MAX_WORDS:
                doc_chunks = sections_to_chunks(pdf_path.name, str(pdf_path), sections)
            else:
                doc_chunks = fixed_size_chunks(full_text, pdf_path.name, str(pdf_path))
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    print(f"Built {len(all_chunks)} chunks before length enforcement")

    final_chunks = enforce_max_length(all_chunks)
    print(f"Final chunk count: {len(final_chunks)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to {OUTPUT_PATH}")

    for c in final_chunks[:6]:
        print("\n---")
        print(c["metadata"])
        print(c["text"][:200], "...")


if __name__ == "__main__":
    main()