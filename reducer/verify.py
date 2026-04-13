import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .protect import detect_protected_spans
from .schemas import Span


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_URL_RE = re.compile(r"https?://[^\s)\]>]+")
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s\"']+|(?:\./|\.\./|/)[A-Za-z0-9_./-]+)")


def _contains_exact(text: str, span: Span) -> bool:
    return span.text in text


def _weighted_anchor_recall(compressed: str, anchors: dict[str, float]) -> float:
    if not anchors:
        return 1.0
    compressed_lower = compressed.lower()
    compressed_compact = re.sub(r"\s+", " ", compressed_lower).strip()
    numerator = sum(
        weight
        for anchor, weight in anchors.items()
        if anchor in compressed_lower
        or re.sub(r"\s+", " ", anchor.lower()).strip() in compressed_compact
    )
    denominator = sum(anchors.values()) or 1.0
    return numerator / denominator


def _negation_preserved(original: str, compressed: str, cfg: Config) -> bool:
    escaped = sorted(
        (re.escape(item) for item in cfg.negation_lexicon), key=len, reverse=True
    )
    pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    original_hits = [item.lower() for item in pattern.findall(original)]
    compressed_hits = [item.lower() for item in pattern.findall(compressed)]
    return original_hits == compressed_hits


def _extract_headings(text: str) -> list[tuple[str, str]]:
    return [(level, title.strip()) for level, title in _HEADING_RE.findall(text)]


def _extract_code_blocks(text: str, cfg: Config) -> list[str]:
    spans = detect_protected_spans(text, cfg)
    return [span.text for span in spans if span.label == "CODE_BLOCK"]


def _structural_preserved(original: str, compressed: str, cfg: Config) -> bool:
    original_headings = _extract_headings(original)
    compressed_headings = _extract_headings(compressed)
    if original_headings and original_headings != compressed_headings:
        return False

    original_code_blocks = _extract_code_blocks(original, cfg)
    compressed_code_blocks = _extract_code_blocks(compressed, cfg)
    if original_code_blocks and original_code_blocks != compressed_code_blocks:
        return False

    if re.search(r"^\s*\d+[.)]\s+", original, re.MULTILINE):
        if not re.search(r"^\s*\d+[.)]\s+", compressed, re.MULTILINE):
            return False

    output_keywords = ("output format", "json", "markdown", "table", "schema")
    if any(keyword in original.lower() for keyword in output_keywords):
        if not any(keyword in compressed.lower() for keyword in output_keywords):
            return False

    if set(_URL_RE.findall(original)) - set(_URL_RE.findall(compressed)):
        return False

    if set(_PATH_RE.findall(original)) - set(_PATH_RE.findall(compressed)):
        return False

    return True


def _lexical_similarity(original: str, compressed: str) -> float:
    if not original.strip() or not compressed.strip():
        return 0.0
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform([original, compressed])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def verify(
    original: str,
    compressed: str,
    spans: list[Span],
    anchors: dict[str, float],
    cfg: Config,
    token_budget: int | None = None,
) -> dict:
    protected_total = len(spans)
    protected_kept = sum(_contains_exact(compressed, span) for span in spans)
    protected_coverage = (
        1.0 if protected_total == 0 else protected_kept / protected_total
    )

    numeric_labels = {"NUMBER", "DATE", "TIME", "SEMVER", "FILE_PATH"}
    numeric_spans = [span for span in spans if span.label in numeric_labels]
    numeric_coverage = (
        1.0 if all(_contains_exact(compressed, span) for span in numeric_spans) else 0.0
    )

    anchor_recall = _weighted_anchor_recall(compressed, anchors)
    negation_preserved = _negation_preserved(original, compressed, cfg)
    structural_preserved = _structural_preserved(original, compressed, cfg)
    lexical_similarity = _lexical_similarity(original, compressed)
    lexical_ok = lexical_similarity >= cfg.lexical_sim_min
    budget_ok = True
    if token_budget is not None:
        from .tokenize import token_count

        budget_ok = token_count(compressed, cfg.tokenizer_model) <= token_budget

    passed = (
        protected_coverage >= 1.0
        and numeric_coverage >= 1.0
        and anchor_recall >= (0.95 if cfg.strict_mode else cfg.anchor_recall_min)
        and negation_preserved
        and structural_preserved
        and budget_ok
        and (lexical_ok if cfg.strict_mode else True)
    )

    return {
        "protected_coverage": round(protected_coverage, 4),
        "numeric_coverage": round(float(numeric_coverage), 4),
        "anchor_recall": round(float(anchor_recall), 4),
        "negation_preserved": negation_preserved,
        "structural_preserved": structural_preserved,
        "budget_ok": budget_ok,
        "lexical_similarity": round(float(lexical_similarity), 4),
        "lexical_ok": lexical_ok,
        "passed": passed,
    }
