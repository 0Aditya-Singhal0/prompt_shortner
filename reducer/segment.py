import re

from .config import Config
from .protect import unit_spans
from .schemas import Span, Unit
from .tokenize import token_count


_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_QUOTE_RE = re.compile(r"^\s*>")
_TABLE_RE = re.compile(r"^\s*\|?.*\|.*\|?.*$")
_KEY_VALUE_RE = re.compile(r"^\s*[A-Za-z0-9_.\-/][^:\n]{0,100}:\s+\S+")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?](?=\s|$))?")


def _clause_guard(sentence: str) -> bool:
    if "`" in sentence:
        return True
    if re.search(r"https?://", sentence):
        return True
    if re.search(r"(?:[A-Za-z]:\\|(?:\./|\.\./|/)[A-Za-z0-9_./-]+)", sentence):
        return True
    if re.search(r"\"[^\"\n]+\"|'[^'\n]+'", sentence):
        return True
    if len(re.findall(r"\b\d+(?:\.\d+)?\b", sentence)) >= 3:
        return True
    if re.search(
        r"\([^)]*(?:must|only|exactly|at least|at most|<=|>=|==|!=)[^)]*\)",
        sentence,
        re.IGNORECASE,
    ):
        return True
    return False


def _split_clauses(sentence: str) -> list[str]:
    if _clause_guard(sentence):
        return [sentence.strip()]

    first_pass = [
        piece.strip() for piece in re.split(r"\s*[;:—–]\s*", sentence) if piece.strip()
    ]
    out: list[str] = []
    for piece in first_pass:
        subparts = [
            x.strip()
            for x in re.split(r"\s+(?:because|so|but|which|that|and)\s+", piece)
            if x.strip()
        ]
        if len(subparts) <= 1:
            out.append(piece)
            continue
        valid = [sub for sub in subparts if len(sub.split()) >= 3]
        if len(valid) == len(subparts):
            out.extend(valid)
        else:
            out.append(piece)
    return out if out else [sentence.strip()]


def _line_offsets(text: str) -> list[tuple[int, str]]:
    offsets: list[tuple[int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        offsets.append((cursor, line))
        cursor += len(line)
    if not offsets and text:
        offsets.append((0, text))
    return offsets


def _non_code_regions(text: str, code_spans: list[Span]) -> list[tuple[int, int]]:
    if not code_spans:
        return [(0, len(text))] if text else []

    regions: list[tuple[int, int]] = []
    cursor = 0
    for span in sorted(code_spans, key=lambda item: (item.start, item.end)):
        if cursor < span.start:
            regions.append((cursor, span.start))
        cursor = max(cursor, span.end)
    if cursor < len(text):
        regions.append((cursor, len(text)))
    return regions


def _add_unit(
    units: list[Unit],
    text: str,
    start: int,
    end: int,
    unit_type: str,
    spans: list[Span],
    cfg: Config,
) -> None:
    body = text.strip()
    if not body:
        return
    unit_id = f"u{len(units)}"
    units.append(
        Unit(
            unit_id=unit_id,
            text=body,
            unit_type=unit_type,
            start=start,
            end=end,
            token_count=token_count(body, cfg.tokenizer_model),
            protected_spans=unit_spans(start, end, spans),
        )
    )


def _segment_paragraph(
    paragraph_text: str,
    paragraph_start: int,
    units: list[Unit],
    spans: list[Span],
    cfg: Config,
) -> None:
    for sentence_match in _SENTENCE_RE.finditer(paragraph_text):
        raw_sentence = sentence_match.group(0)
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        sentence_start = paragraph_start + sentence_match.start()
        sentence_end = paragraph_start + sentence_match.end()

        clauses = _split_clauses(sentence)
        if len(clauses) <= 1:
            _add_unit(
                units, sentence, sentence_start, sentence_end, "SENTENCE", spans, cfg
            )
            continue

        cursor = sentence_start
        for clause in clauses:
            local_index = paragraph_text.find(
                clause, cursor - paragraph_start, sentence_end - paragraph_start
            )
            if local_index < 0:
                local_index = cursor - paragraph_start
            clause_start = paragraph_start + local_index
            clause_end = clause_start + len(clause)
            _add_unit(units, clause, clause_start, clause_end, "CLAUSE", spans, cfg)
            cursor = clause_end


def segment(text: str, spans: list[Span], cfg: Config) -> list[Unit]:
    units: list[Unit] = []

    code_spans = [span for span in spans if span.label == "CODE_BLOCK"]
    for code_span in code_spans:
        _add_unit(
            units,
            text[code_span.start : code_span.end],
            code_span.start,
            code_span.end,
            "CODE_BLOCK",
            spans,
            cfg,
        )

    for region_start, region_end in _non_code_regions(text, code_spans):
        if region_start >= region_end:
            continue

        paragraph_start: int | None = None
        paragraph_end = region_start

        def flush_paragraph() -> None:
            nonlocal paragraph_start, paragraph_end
            if paragraph_start is None:
                return
            chunk = text[paragraph_start:paragraph_end]
            _segment_paragraph(chunk, paragraph_start, units, spans, cfg)
            paragraph_start = None

        chunk = text[region_start:region_end]
        for rel_start, line in _line_offsets(chunk):
            line_start = region_start + rel_start
            line_end = line_start + len(line)

            bare = line.strip()
            if not bare:
                flush_paragraph()
                continue

            if _HEADING_RE.match(line):
                flush_paragraph()
                _add_unit(units, line, line_start, line_end, "SENTENCE", spans, cfg)
                continue
            if _LIST_RE.match(line):
                flush_paragraph()
                _add_unit(units, line, line_start, line_end, "LIST_ITEM", spans, cfg)
                continue
            if _QUOTE_RE.match(line):
                flush_paragraph()
                _add_unit(units, line, line_start, line_end, "QUOTE_BLOCK", spans, cfg)
                continue
            if _TABLE_RE.match(line) and line.count("|") >= 2:
                flush_paragraph()
                _add_unit(
                    units,
                    line,
                    line_start,
                    line_end,
                    "KEY_VALUE_LINE",
                    spans,
                    cfg,
                )
                continue
            if _KEY_VALUE_RE.match(line):
                flush_paragraph()
                _add_unit(
                    units,
                    line,
                    line_start,
                    line_end,
                    "KEY_VALUE_LINE",
                    spans,
                    cfg,
                )
                continue

            if paragraph_start is None:
                paragraph_start = line_start
            paragraph_end = line_end

        flush_paragraph()

    units.sort(key=lambda unit: (unit.start, unit.end, unit.unit_id))
    deduped: list[Unit] = []
    seen: set[tuple[int, int, str, str]] = set()
    for unit in units:
        key = (unit.start, unit.end, unit.unit_type, unit.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit)
    for index, unit in enumerate(deduped):
        unit.unit_id = f"u{index}"
    return deduped
