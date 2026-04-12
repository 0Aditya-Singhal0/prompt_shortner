import re

from .config import Config
from .schemas import Span


PATTERNS = {
    "CODE_BLOCK": re.compile(r"```[\s\S]*?```", re.MULTILINE),
    "INLINE_CODE": re.compile(r"`[^`\n]+`"),
    "URL": re.compile(r"https?://\S+"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "FILE_PATH": re.compile(r"(?:[A-Za-z]:\\[^\s]+|/[\w./-]+)"),
    "SEMVER": re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),
    "SHA": re.compile(r"\b[a-f0-9]{7,40}\b"),
    "CLI_FLAG": re.compile(r"(?<!\w)--?[a-zA-Z][\w-]*\b"),
    "ENV_VAR": re.compile(r"\b[A-Z_][A-Z0-9_]{2,}\b"),
    "NUMBER": re.compile(r"\b\d+(?:\.\d+)?(?:%|ms|s|sec|min|h|MB|GB|KB|x)?\b"),
    "DATE": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "TIME": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    "QUOTE": re.compile(r"\"[^\"]+\"|'[^'\n]+'"),
    "IDENTIFIER": re.compile(
        r"\b(?:[a-z]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][A-Za-z0-9]+)\b"
    ),
}


def _merge_overlaps(spans: list[Span]) -> list[Span]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    merged = [spans[0]]
    for span in spans[1:]:
        last = merged[-1]
        if span.start <= last.end:
            if span.end > last.end:
                merged[-1] = Span(last.start, span.end, last.label, last.text)
        else:
            merged.append(span)
    return merged


def detect_protected_spans(text: str, cfg: Config) -> list[Span]:
    spans: list[Span] = []
    for label, pat in PATTERNS.items():
        for m in pat.finditer(text):
            spans.append(Span(m.start(), m.end(), label, m.group(0)))

    negation = re.compile(
        "|".join(re.escape(x) for x in cfg.negation_lexicon), re.IGNORECASE
    )
    for m in negation.finditer(text):
        spans.append(Span(m.start(), m.end(), "NEGATION", m.group(0)))
    return _merge_overlaps(spans)


def unit_spans(unit_start: int, unit_end: int, spans: list[Span]) -> list[Span]:
    out = []
    for s in spans:
        if s.end <= unit_start or s.start >= unit_end:
            continue
        out.append(
            Span(
                max(s.start, unit_start),
                min(s.end, unit_end),
                s.label,
                s.text,
            )
        )
    return out
