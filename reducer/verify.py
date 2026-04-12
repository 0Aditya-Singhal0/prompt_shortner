import re

from .config import Config
from .schemas import Span


def _contains_exact(text: str, span: Span) -> bool:
    return span.text in text


def verify(
    original: str,
    compressed: str,
    spans: list[Span],
    anchors: dict[str, float],
    cfg: Config,
) -> dict:
    protected_total = len(spans)
    protected_ok = sum(_contains_exact(compressed, s) for s in spans)
    protected_coverage = 1.0 if protected_total == 0 else protected_ok / protected_total

    numeric_spans = [
        s for s in spans if s.label in {"NUMBER", "DATE", "TIME", "SEMVER", "FILE_PATH"}
    ]
    numeric_ok = all(_contains_exact(compressed, s) for s in numeric_spans)

    denom = sum(anchors.values()) or 1.0
    numer = sum(w for a, w in anchors.items() if a in compressed.lower())
    anchor_recall = numer / denom

    neg = re.compile(
        r"\b(?:must not|do not|never|cannot|without|no more than)\b", re.IGNORECASE
    )
    negation_preserved = len(neg.findall(original)) == len(neg.findall(compressed))

    structural = True
    if re.search(r"^\s*\d+[.)]\s+", original, re.MULTILINE):
        structural = bool(re.search(r"^\s*\d+[.)]\s+", compressed, re.MULTILINE))
    if any(
        k in original.lower() for k in ("output format", "json", "markdown", "table")
    ):
        structural = structural and any(
            k in compressed.lower()
            for k in ("output format", "json", "markdown", "table")
        )

    passed = (
        protected_coverage >= 1.0
        and numeric_ok
        and anchor_recall >= (0.95 if cfg.strict_mode else cfg.anchor_recall_min)
        and negation_preserved
        and structural
    )
    return {
        "protected_coverage": round(protected_coverage, 4),
        "numeric_coverage": 1.0 if numeric_ok else 0.0,
        "anchor_recall": float(round(anchor_recall, 4)),
        "negation_preserved": negation_preserved,
        "structural_preserved": structural,
        "passed": passed,
    }
