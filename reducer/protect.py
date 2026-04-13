import re
from dataclasses import dataclass

from .config import Config
from .schemas import Span


@dataclass(frozen=True)
class _PatternSpec:
    label: str
    pattern: re.Pattern[str]
    priority: int


_PATTERNS: tuple[_PatternSpec, ...] = (
    _PatternSpec("CODE_BLOCK", re.compile(r"(```|~~~)[\s\S]*?\1", re.MULTILINE), 100),
    _PatternSpec("INLINE_CODE", re.compile(r"`[^`\n]+`"), 96),
    _PatternSpec("URL", re.compile(r"https?://[^\s)\]>]+"), 92),
    _PatternSpec(
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        90,
    ),
    _PatternSpec(
        "FILE_PATH",
        re.compile(r"(?:[A-Za-z]:\\[^\s\"']+|(?:\./|\.\./|/)[A-Za-z0-9_./-]+)"),
        88,
    ),
    _PatternSpec("SEMVER", re.compile(r"\bv?\d+\.\d+(?:\.\d+){0,2}\b"), 86),
    _PatternSpec("SHA", re.compile(r"\b[a-f0-9]{7,40}\b"), 84),
    _PatternSpec("ISSUE_ID", re.compile(r"\B#[0-9]{1,7}\b"), 82),
    _PatternSpec("CLI_FLAG", re.compile(r"(?<!\w)--?[A-Za-z][A-Za-z0-9-]*\b"), 80),
    _PatternSpec("ENV_VAR", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b"), 78),
    _PatternSpec("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), 76),
    _PatternSpec("TIME", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), 74),
    _PatternSpec(
        "NUMBER",
        re.compile(r"\b\d+(?:\.\d+)?(?:%|ms|s|sec|min|h|MB|GB|KB|x)?\b"),
        72,
    ),
    _PatternSpec("QUOTE", re.compile(r"\"[^\"\n]+\"|'[^'\n]+'"), 70),
    _PatternSpec(
        "IDENTIFIER",
        re.compile(
            r"\b(?:[a-z]+(?:_[a-z0-9]+)+|"
            r"[a-z]+(?:[A-Z][a-z0-9]+)+|"
            r"[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|"
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b"
        ),
        68,
    ),
    _PatternSpec(
        "OUTPUT_FORMAT",
        re.compile(
            r"\b(?:output\s+format|return\s+(?:json|markdown|table)|"
            r"as\s+(?:json|markdown|table)|json\s+schema|markdown\s+table)\b",
            re.IGNORECASE,
        ),
        66,
    ),
)


def _spans_overlap(a: Span, b: Span) -> bool:
    return not (a.end <= b.start or b.end <= a.start)


def _iter_pattern_spans(text: str) -> list[tuple[Span, int]]:
    out: list[tuple[Span, int]] = []
    for spec in _PATTERNS:
        for match in spec.pattern.finditer(text):
            out.append(
                (
                    Span(match.start(), match.end(), spec.label, match.group(0)),
                    spec.priority,
                )
            )
    return out


def _iter_negation_spans(text: str, cfg: Config) -> list[tuple[Span, int]]:
    escaped = sorted(
        (re.escape(item) for item in cfg.negation_lexicon), key=len, reverse=True
    )
    pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return [
        (Span(m.start(), m.end(), "NEGATION", m.group(0)), 94)
        for m in pattern.finditer(text)
    ]


def _select_non_overlapping(candidates: list[tuple[Span, int]]) -> list[Span]:
    ranked = sorted(
        candidates,
        key=lambda item: (-item[1], -(item[0].end - item[0].start), item[0].start),
    )
    selected: list[Span] = []
    for span, _priority in ranked:
        if any(_spans_overlap(span, kept) for kept in selected):
            continue
        selected.append(span)
    selected.sort(key=lambda s: (s.start, s.end))
    return selected


def detect_protected_spans(text: str, cfg: Config) -> list[Span]:
    candidates = _iter_pattern_spans(text)
    candidates.extend(_iter_negation_spans(text, cfg))
    return _select_non_overlapping(candidates)


def unit_spans(unit_start: int, unit_end: int, spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for span in spans:
        if span.end <= unit_start or span.start >= unit_end:
            continue
        start = max(span.start, unit_start)
        end = min(span.end, unit_end)
        out.append(Span(start, end, span.label, span.text))
    return out
