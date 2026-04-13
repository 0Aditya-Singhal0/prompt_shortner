from collections import Counter

import yake
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .schemas import Span, Unit
from .tokenize import lexical_tokens


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
}

_IMPERATIVE_HEADS = {
    "add",
    "build",
    "check",
    "compress",
    "create",
    "debug",
    "delete",
    "ensure",
    "explain",
    "extract",
    "fix",
    "generate",
    "implement",
    "keep",
    "optimize",
    "preserve",
    "refactor",
    "remove",
    "return",
    "run",
    "segment",
    "split",
    "update",
    "verify",
    "write",
}


def _clean_anchor(text: str) -> str:
    cleaned = text.strip().strip("`\"'()[]{}.,:;!?")
    return " ".join(cleaned.lower().split())


def _bounded_add(anchor_weights: dict[str, float], anchor: str, value: float) -> None:
    if not anchor:
        return
    anchor_weights[anchor] = max(anchor_weights.get(anchor, 0.0), value)


def _extract_imperative_targets(units: list[Unit]) -> list[str]:
    targets: list[str] = []
    leadins = {"can", "could", "would", "please", "i", "want", "you", "to", "need"}
    for unit in units:
        tokens = lexical_tokens(unit.text)
        if len(tokens) < 2:
            continue
        head_index = None
        for index, token in enumerate(tokens[:8]):
            if token in _IMPERATIVE_HEADS:
                head_index = index
                break
            if token not in leadins:
                break
        if head_index is None:
            continue
        targets.append(tokens[head_index])
        for token in tokens[head_index + 1 : head_index + 5]:
            if token not in _STOPWORDS and token not in leadins and len(token) > 0:
                targets.append(token)
    return targets


def _extract_final_question_nouns(text: str) -> list[str]:
    questions = [chunk.strip() for chunk in text.split("?") if chunk.strip()]
    if not text.strip().endswith("?") or not questions:
        return []
    last = questions[-1]
    return [
        token
        for token in lexical_tokens(last)
        if token not in _STOPWORDS and len(token) > 2
    ]


def _extract_heading_tokens(units: list[Unit]) -> list[str]:
    out: list[str] = []
    for unit in units:
        content = unit.text.lstrip()
        if content.startswith("#") or unit.unit_type == "LIST_ITEM":
            out.extend(
                token for token in lexical_tokens(content) if token not in _STOPWORDS
            )
    return out


def extract_anchors(
    text: str, units: list[Unit], spans: list[Span], cfg: Config
) -> dict[str, float]:
    weights: dict[str, float] = {}
    freq = Counter(lexical_tokens(text))
    imperative_targets = _extract_imperative_targets(units)
    imperative_set = set(imperative_targets)
    protected_tokens = {
        token
        for span in spans
        for token in lexical_tokens(_clean_anchor(span.text))
        if token
    }
    noise_tokens = set(
        lexical_tokens(
            " ".join(
                cfg.boilerplate_terms
                + cfg.output_hedges
                + cfg.output_meta_phrases
                + cfg.output_pleasantries
            )
        )
    )

    for span in spans:
        anchor = _clean_anchor(span.text)
        if not anchor:
            continue
        if span.label in {"IDENTIFIER", "INLINE_CODE", "CODE_BLOCK"}:
            type_bonus = 1.5
        elif span.label in {
            "NUMBER",
            "DATE",
            "TIME",
            "SEMVER",
            "FILE_PATH",
            "URL",
            "CLI_FLAG",
            "ENV_VAR",
        }:
            type_bonus = 1.2
        elif span.label in {"NEGATION", "OUTPUT_FORMAT"}:
            type_bonus = 1.3
        else:
            type_bonus = 1.0
        _bounded_add(weights, anchor, type_bonus)

    yake_keywords = yake.KeywordExtractor(
        lan="en", n=3, top=cfg.anchor_top_yake
    ).extract_keywords(text)
    for rank, (phrase, score) in enumerate(yake_keywords, start=1):
        anchor = _clean_anchor(str(phrase))
        if not anchor:
            continue
        phrase_tokens = lexical_tokens(anchor)
        if not phrase_tokens:
            continue
        if phrase_tokens and all(
            token in noise_tokens or token in _STOPWORDS for token in phrase_tokens
        ):
            continue
        if not any(
            token in protected_tokens
            or token in imperative_set
            or freq.get(token, 0) >= 2
            for token in phrase_tokens
        ):
            continue
        salience = 1.0 / (1.0 + float(score))
        _bounded_add(weights, anchor, 0.4 * salience + 0.8 / (1.0 + rank))

    docs = [unit.text for unit in units if unit.text.strip()]
    if docs:
        vectorizer = TfidfVectorizer(
            lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=1
        )
        vectorizer.fit_transform(docs)
        names = vectorizer.get_feature_names_out().tolist()
        idf_vals = vectorizer.idf_.tolist()
        ranked = sorted(zip(names, idf_vals), key=lambda item: item[1], reverse=True)
        max_idf = max(idf_vals) if idf_vals else 1.0
        for token, idf in ranked[: cfg.anchor_top_idf]:
            if token in _STOPWORDS or len(token) < 2:
                continue
            if token in noise_tokens:
                continue
            if (
                token not in protected_tokens
                and token not in imperative_set
                and freq.get(token, 0) < 2
            ):
                continue
            _bounded_add(weights, token, 0.6 * (idf / max_idf))

    for token in imperative_targets:
        if token in noise_tokens:
            continue
        _bounded_add(weights, token, 0.7)

    for token in _extract_final_question_nouns(text):
        if token in noise_tokens:
            continue
        _bounded_add(weights, token, 0.6)

    for token in _extract_heading_tokens(units):
        if token in noise_tokens:
            continue
        _bounded_add(weights, token, 0.6)

    repeated_entities = [
        token
        for token, count in freq.items()
        if count >= 2 and len(token) > 2 and token not in _STOPWORDS
    ]
    for token in repeated_entities:
        if token in noise_tokens:
            continue
        _bounded_add(weights, token, 0.5)

    for anchor in list(weights):
        repeat_bonus = min(0.6, 0.15 * freq.get(anchor, 0))
        weights[anchor] += repeat_bonus

    ranked_anchors = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    cap = max(8, min(cfg.anchor_max_terms, max(8, len(units) * 3)))
    capped = [(anchor, weight) for anchor, weight in ranked_anchors if weight >= 0.5][
        :cap
    ]
    return {anchor: float(weight) for anchor, weight in capped}
