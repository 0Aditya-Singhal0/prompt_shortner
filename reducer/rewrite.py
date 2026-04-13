import re

from .config import Config
from .schemas import Span, Unit


_DISCOURSE_PATTERNS = (
    r"\b[Cc]an you please\b",
    r"\b[Ii] want you to\b",
    r"\b[Bb]asically\b",
    r"\b[Ii]t would be great if you could\b",
)

_MODIFIER_PATTERN = re.compile(
    r"\b(?:very|really|quite|highly|extremely|detailed)\b", re.IGNORECASE
)


def rewrite_eligible(unit: Unit, cfg: Config) -> bool:
    if unit.hard_keep:
        return False
    if unit.features.get("protected_mass", 1.0) >= cfg.rewrite_protected_mass_max:
        return False
    labels = {span.label for span in unit.protected_spans}
    blocked_labels = {
        "CODE_BLOCK",
        "INLINE_CODE",
        "FILE_PATH",
        "SEMVER",
        "QUOTE",
        "NUMBER",
        "NEGATION",
        "URL",
        "OUTPUT_FORMAT",
    }
    if labels & blocked_labels:
        return False
    return unit.token_count >= 12


def _mask_spans(text: str, spans: list[Span]) -> tuple[str, dict[str, str]]:
    if not spans:
        return text, {}
    pieces: list[str] = []
    mapping: dict[str, str] = {}
    cursor = 0
    for index, span in enumerate(
        sorted(spans, key=lambda item: (item.start, item.end))
    ):
        placeholder = f"__P{index}__"
        pieces.append(text[cursor : span.start])
        pieces.append(placeholder)
        mapping[placeholder] = text[span.start : span.end]
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), mapping


def _unmask_spans(text: str, mapping: dict[str, str]) -> str:
    out = text
    for placeholder, original in mapping.items():
        out = out.replace(placeholder, original)
    return out


def safe_rewrite(text: str, spans: list[Span] | None = None) -> str:
    masked, mapping = _mask_spans(text, spans or [])

    out = masked
    for pattern in _DISCOURSE_PATTERNS:
        out = re.sub(pattern, "", out)

    out = re.sub(
        r"\b[Cc]an you\s+explain\s+(.+?)\s+and also\s+tell me\s+(.+?)(?:[.?!]|$)",
        r"Explain \1; tell me \2.",
        out,
    )
    out = re.sub(r"\b[Ii] need you to provide\b", "Provide", out)
    out = re.sub(r"\band also\b", ", ", out, flags=re.IGNORECASE)
    out = _MODIFIER_PATTERN.sub("", out)
    out = re.sub(r"\b[Ii] also want\b", "", out)

    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*,", ", ", out)
    out = re.sub(r"\s+([.?!:;])", r"\1", out)
    out = out.strip(" ,;")
    if out and out[-1] not in ".?!":
        out += "."

    out = _unmask_spans(out, mapping)
    return out or text
