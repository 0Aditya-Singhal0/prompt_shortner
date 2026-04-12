import re

from .protect import unit_spans
from .schemas import Span, Unit
from .tokenize import token_count


LIST_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
SENT_RE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)


def _is_protected_sentence(text: str) -> bool:
    guards = [r"`", r"https?://", r"[A-Za-z]:\\", r"/[\w./-]+", r'"[^"]+"', r"\d{3,}"]
    return any(re.search(g, text) for g in guards)


def segment(text: str, spans: list[Span]) -> list[Unit]:
    units: list[Unit] = []
    uid = 0
    for block_match in re.finditer(r"```[\s\S]*?```", text):
        s, e = block_match.span()
        utext = text[s:e]
        units.append(
            Unit(
                f"u{uid}",
                utext,
                "CODE_BLOCK",
                s,
                e,
                token_count(utext),
                unit_spans(s, e, spans),
            )
        )
        uid += 1

    covered = [(u.start, u.end) for u in units]

    def covered_by_code(start: int, end: int) -> bool:
        return any(start >= cs and end <= ce for cs, ce in covered)

    scan = 0
    for line in text.splitlines(keepends=True):
        pos = text.find(line, scan)
        if pos == -1:
            pos = scan
        scan = pos + len(line)
        if covered_by_code(pos, pos + len(line)):
            continue
        if LIST_RE.match(line):
            s, e = pos, pos + len(line)
            utext = line.strip()
            if utext:
                units.append(
                    Unit(
                        f"u{uid}",
                        utext,
                        "LIST_ITEM",
                        s,
                        e,
                        token_count(utext),
                        unit_spans(s, e, spans),
                    )
                )
                uid += 1

    for m in SENT_RE.finditer(text):
        s, e = m.span()
        if covered_by_code(s, e):
            continue
        sentence = m.group(0).strip()
        if not sentence or LIST_RE.match(sentence):
            continue
        if _is_protected_sentence(sentence):
            parts = [sentence]
        else:
            parts = [
                p.strip() for p in re.split(r";|:\s+|\s+and\s+", sentence) if p.strip()
            ]
        cursor = s
        for part in parts:
            pi = text.find(part, cursor, e)
            if pi == -1:
                pi = cursor
            pj = pi + len(part)
            units.append(
                Unit(
                    f"u{uid}",
                    part,
                    "CLAUSE" if len(parts) > 1 else "SENTENCE",
                    pi,
                    pj,
                    token_count(part),
                    unit_spans(pi, pj, spans),
                )
            )
            uid += 1
            cursor = pj

    units.sort(key=lambda u: (u.start, u.end))
    seen = set()
    deduped = []
    for u in units:
        key = (u.start, u.end, u.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(u)
    return deduped
