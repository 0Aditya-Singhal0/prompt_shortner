from functools import lru_cache
import re

import nltk

from .config import Config
from .protect import detect_protected_spans
from .schemas import Span
from .tokenize import lexical_tokens


_PLACEHOLDER_RE = re.compile(r"__P\d+__")
_URL_RE = re.compile(r"https?://[^\s)\]>]+")
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s\"']+|(?:\./|\.\./|/)[A-Za-z0-9_./-]+)")
_PAREN_RE = re.compile(r"\(([^()]+)\)")
_LEADING_SUBORDINATE_RE = re.compile(
    r"^\s*(?:because|since|although|while|if)\b[^,]{0,120},\s*(.+)$",
    re.IGNORECASE,
)
_PRUNABLE_PAREN_HINTS = (
    "for example",
    "e.g.",
    "i think",
    "usually",
    "generally",
    "maybe",
    "probably",
)
_POS_DROP_TAGS = {"RB", "RBR", "RBS"}
_ADJECTIVE_TAGS = {"JJ", "JJR", "JJS"}
_DISCOURSE_HEADS = {"however", "therefore", "additionally", "moreover"}


@lru_cache(maxsize=1)
def _pos_tagger_available() -> bool:
    tagger_paths = (
        "taggers/averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger",
    )
    for path in tagger_paths:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False


def _pos_tag_tokens(text: str) -> list[tuple[str, str]]:
    tokens = nltk.tokenize.wordpunct_tokenize(text)
    if not tokens:
        return []
    if not _pos_tagger_available():
        return [(token, "") for token in tokens]
    try:
        return [(token, tag) for token, tag in nltk.pos_tag(tokens)]
    except LookupError:
        return [(token, "") for token in tokens]


def _join_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    out = " ".join(tokens)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([([{])\s+", r"\1", out)
    out = re.sub(r"\s+([)\]}])", r"\1", out)
    return out.strip()


def _prune_by_pos(text: str, cfg: Config) -> str:
    if not cfg.output_use_pos_pruning:
        return text
    tagged = _pos_tag_tokens(text)
    if not tagged:
        return text

    modifiers = set(cfg.output_prunable_modifiers)
    adjectives = set(cfg.output_prunable_adjectives)
    negations = {item.lower() for item in cfg.negation_lexicon}
    kept: list[str] = []
    for token, tag in tagged:
        lower = token.lower()
        if lower in negations:
            kept.append(token)
            continue
        if tag in _POS_DROP_TAGS and lower in modifiers:
            continue
        if tag in _ADJECTIVE_TAGS and lower in adjectives:
            continue
        if lower in _DISCOURSE_HEADS and tag in _POS_DROP_TAGS | {"CC"}:
            continue
        kept.append(token)
    return _join_tokens(kept)


def _prune_by_dependency(text: str, cfg: Config) -> str:
    if not cfg.output_use_dependency_pruning:
        return text

    out = re.sub(
        r"^\s*(?:however|therefore|additionally|moreover)\s*,?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    match = _LEADING_SUBORDINATE_RE.match(out)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            out = candidate
    return out.strip()


def detect_protected_output_spans(text: str, cfg: Config) -> list[Span]:
    return detect_protected_spans(text, cfg)


def _mask_spans(text: str, spans: list[Span]) -> tuple[str, dict[str, str]]:
    if not spans:
        return text, {}
    parts: list[str] = []
    mapping: dict[str, str] = {}
    cursor = 0
    for index, span in enumerate(
        sorted(spans, key=lambda item: (item.start, item.end))
    ):
        placeholder = f"__P{index}__"
        parts.append(text[cursor : span.start])
        parts.append(placeholder)
        mapping[placeholder] = text[span.start : span.end]
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts), mapping


def _unmask_spans(text: str, mapping: dict[str, str]) -> str:
    out = text
    for placeholder, original in mapping.items():
        out = out.replace(placeholder, original)
    return out


def discourse_fluff_score(sentence: str, cfg: Config) -> float:
    tokens = lexical_tokens(sentence)
    if not tokens:
        return 0.0
    lower = sentence.lower()
    hedge_density = sum(lower.count(term) for term in cfg.output_hedges) / len(tokens)
    pleasantry_density = sum(
        lower.count(term) for term in cfg.output_pleasantries
    ) / len(tokens)
    meta_density = sum(lower.count(term) for term in cfg.output_meta_phrases) / len(
        tokens
    )
    return 0.45 * hedge_density + 0.35 * pleasantry_density + 0.20 * meta_density


def _clause_guard(sentence: str, cfg: Config) -> bool:
    lower = sentence.lower()
    if _PLACEHOLDER_RE.search(sentence):
        return True
    if "`" in sentence:
        return True
    if _URL_RE.search(sentence):
        return True
    if _PATH_RE.search(sentence):
        return True
    if re.search(r'"[^"\n]+"|\'[^\'\n]+\'', sentence):
        return True
    if len(re.findall(r"\b\d+(?:\.\d+)?\b", sentence)) >= 2:
        return True
    if any(term in lower for term in cfg.negation_lexicon):
        return True
    return False


def split_into_clauses(sentence: str, spans: list[Span], cfg: Config) -> list[str]:
    del spans
    if _clause_guard(sentence, cfg):
        stripped = sentence.strip()
        return [stripped] if stripped else [sentence]

    clauses = [sentence]

    first_pass: list[str] = []
    for chunk in clauses:
        first_pass.extend(
            [
                piece.strip()
                for piece in re.split(r"\s*[;:—–]\s*", chunk)
                if piece.strip()
            ]
        )

    second_pass: list[str] = []
    for chunk in first_pass:
        pieces = [
            piece.strip()
            for piece in re.split(r"\s+(?:because|so|but|which)\s+", chunk)
            if piece.strip()
        ]
        if len(pieces) <= 1:
            second_pass.append(chunk)
            continue
        if all(len(piece.split()) >= 4 for piece in pieces):
            second_pass.extend(pieces)
        else:
            second_pass.append(chunk)
    return second_pass if second_pass else [sentence]


def _parenthetical_is_prunable(content: str, cfg: Config) -> bool:
    inner = content.strip()
    if not inner:
        return True

    lower = inner.lower()
    if _PLACEHOLDER_RE.search(inner):
        return False
    if _URL_RE.search(inner) or _PATH_RE.search(inner):
        return False
    if re.search(r"\d", inner):
        return False
    if re.search(r"(?:<=|>=|==|!=|=>|->)", inner):
        return False
    if any(term in lower for term in cfg.negation_lexicon):
        return False
    if any(hint in lower for hint in _PRUNABLE_PAREN_HINTS):
        return True
    return len(inner.split()) <= 2


def _prune_parentheticals(text: str, cfg: Config) -> str:
    return _PAREN_RE.sub(
        lambda match: ""
        if _parenthetical_is_prunable(match.group(1), cfg)
        else match.group(0),
        text,
    )


def prune_clause(clause: str, spans: list[Span], cfg: Config) -> str:
    del spans
    text = clause
    for pattern in cfg.output_fluff_patterns:
        text = re.sub(pattern, "", text)

    text = _prune_parentheticals(text, cfg)
    text = _prune_by_dependency(text, cfg)
    text = _prune_by_pos(text, cfg)
    modifier_pattern = (
        r"\b(?:"
        + "|".join(re.escape(term) for term in cfg.output_prunable_modifiers)
        + r")\b"
    )
    text = re.sub(modifier_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:it is important to note that|you should understand that)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip(" ,;")


def allow_fragment(clause: str, cfg: Config, output_keep_ratio: float) -> bool:
    if output_keep_ratio > cfg.output_fragment_allow_threshold:
        return False
    if len(clause.split()) < 3:
        return False
    if _PLACEHOLDER_RE.search(clause):
        return False
    return True


def _fragmentize(clause: str) -> str:
    out = clause.strip()
    out = re.sub(r"^(?:the|a|an)\s+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^(?:this|it|there)\s+is\s+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^(?:the\s+)?issue\s+is\s+", "Issue: ", out, flags=re.IGNORECASE)
    out = re.sub(r"^(?:this\s+)?causes\s+", "Causes ", out, flags=re.IGNORECASE)
    return out.strip()


def output_risk_score(clause: str, cfg: Config) -> float:
    tokens = lexical_tokens(clause)
    if not tokens:
        return 0.0
    lower = clause.lower()

    negation_hits = sum(lower.count(term) for term in cfg.negation_lexicon)
    n_score = min(1.0, negation_hits / max(1, len(tokens)))

    sequence_hits = 0
    if re.search(r"^\s*\d+[.)]\s+", clause):
        sequence_hits += 1
    if re.search(r"\b(?:first|then|next|finally|step)\b", lower):
        sequence_hits += 1
    q_score = min(1.0, sequence_hits / 2.0)

    warning_hits = sum(1 for term in cfg.output_risk_terms if term in lower)
    w_score = min(1.0, warning_hits / 3.0)

    return 0.5 * n_score + 0.3 * q_score + 0.2 * w_score


def _line_is_structured(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return True
    if stripped.startswith(("#", "-", "*", ">")):
        return True
    if re.match(r"^\d+[.)]\s+", stripped):
        return True
    if stripped.count("|") >= 2:
        return True
    if re.match(r"^[\[{]", stripped) and stripped.endswith(("]", "}")):
        return True
    return False


def _sentence_protected_mass(masked_sentence: str) -> float:
    placeholders = _PLACEHOLDER_RE.findall(masked_sentence)
    if not placeholders:
        return 0.0
    text_len = max(1, len(masked_sentence))
    protected_len = sum(len(placeholder) for placeholder in placeholders)
    return min(1.0, protected_len / text_len)


def normalize_output(
    text: str, cfg: Config, output_keep_ratio: float | None = None
) -> str:
    if not text.strip() or not cfg.use_output_normalizer:
        return text

    keep_ratio = (
        output_keep_ratio if output_keep_ratio is not None else cfg.output_keep_ratio()
    )
    spans = detect_protected_output_spans(text, cfg)
    masked, mapping = _mask_spans(text, spans)

    normalized_lines: list[str] = []
    for line in masked.splitlines():
        if not line.strip():
            normalized_lines.append("")
            continue
        if _line_is_structured(line):
            normalized_lines.append(line.rstrip())
            continue

        chunks = [
            chunk.strip()
            for chunk in re.split(r"(?<=[.!?])\s+", line.strip())
            if chunk.strip()
        ]
        if not chunks:
            normalized_lines.append(line.rstrip())
            continue

        reduced_chunks: list[str] = []
        for chunk in chunks:
            if _sentence_protected_mass(chunk) >= cfg.output_protected_mass_bypass:
                reduced_chunks.append(chunk)
                continue
            if output_risk_score(chunk, cfg) >= cfg.output_risk_bypass_threshold:
                reduced_chunks.append(chunk)
                continue

            candidate = chunk
            if (
                discourse_fluff_score(candidate, cfg)
                >= cfg.output_fluff_strip_threshold
            ):
                candidate = prune_clause(candidate, [], cfg)

            clauses = split_into_clauses(candidate, [], cfg)
            for clause in clauses:
                if output_risk_score(clause, cfg) >= cfg.output_risk_bypass_threshold:
                    reduced_chunks.append(clause)
                    continue
                pruned = prune_clause(clause, [], cfg)
                if allow_fragment(pruned, cfg, keep_ratio):
                    pruned = _fragmentize(pruned)
                if pruned:
                    reduced_chunks.append(pruned)

        if not reduced_chunks:
            normalized_lines.append(line.rstrip())
            continue

        normalized_line = "; ".join(reduced_chunks)
        normalized_line = re.sub(r"\s{2,}", " ", normalized_line).strip()
        normalized_lines.append(normalized_line)

    normalized = "\n".join(normalized_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = _unmask_spans(normalized, mapping)
    normalized = re.sub(r"[^\S\n]{2,}", " ", normalized)
    normalized = re.sub(r"[^\S\n]+([,.;:!?])", r"\1", normalized)
    return normalized.strip()
