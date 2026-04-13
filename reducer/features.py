import re

import networkx as nx
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .schemas import Unit
from .tokenize import lexical_tokens, shingle_set, token_count


_STOPWORD_SET = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
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
    "we",
    "with",
    "you",
    "your",
}

_HEDGE_SET = {"maybe", "perhaps", "possibly", "somewhat", "likely", "probably"}


def _minmax(values: list[float], eps: float) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= eps:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo + eps) for value in values]


def _protected_mass(unit: Unit, cfg: Config) -> float:
    protected_tokens = sum(
        token_count(span.text, cfg.tokenizer_model) for span in unit.protected_spans
    )
    return min(1.0, protected_tokens / max(unit.token_count, 1))


def _structure_score(unit: Unit, index: int, total: int) -> float:
    text = unit.text.strip()
    lower = text.lower()
    score = 0.0
    if re.match(r"^\d+[.)]\s+", text):
        score += 0.30
    if re.match(r"^[-*+]\s+", text):
        score += 0.20
    if ":" in text and len(text.split(":")) >= 2:
        score += 0.20
    if any(term in lower for term in ("must", "never", "only", "exactly")):
        score += 0.35
    if index == total - 1 and "?" in text:
        score += 0.30
    if any(
        term in lower
        for term in ("output", "format", "json", "table", "markdown", "return")
    ):
        score += 0.25
    if text.startswith("#"):
        score += 0.20
    return min(1.0, score)


def _exactness_score(unit: Unit) -> float:
    identifiers = sum(span.label == "IDENTIFIER" for span in unit.protected_spans)
    numbers = sum(
        span.label in {"NUMBER", "DATE", "TIME", "SEMVER"}
        for span in unit.protected_spans
    )
    quoted = sum(span.label == "QUOTE" for span in unit.protected_spans)
    return min(1.0, (identifiers + numbers + quoted) / 3.0)


def _boilerplate_penalty(unit: Unit, cfg: Config) -> float:
    text = unit.text.lower()
    tokens = lexical_tokens(text)
    if not tokens:
        return 0.0
    boiler_density = sum(text.count(term) for term in cfg.boilerplate_terms) / len(
        tokens
    )
    stop_ratio = sum(token in _STOPWORD_SET for token in tokens) / len(tokens)
    hedge_ratio = sum(token in _HEDGE_SET for token in tokens) / len(tokens)
    return min(1.0, 0.5 * boiler_density + 0.3 * stop_ratio + 0.2 * hedge_ratio)


def _position_prior(unit: Unit, index: int, total: int) -> float:
    lower = unit.text.lower()
    if index == total - 1 and "?" in unit.text:
        return 0.30
    if index == 0 and any(
        word in lower
        for word in ("task", "need", "build", "implement", "fix", "create")
    ):
        return 0.20
    return 0.0


def compute_features(
    units: list[Unit], anchors: dict[str, float], cfg: Config
) -> list[Unit]:
    if not units:
        return units

    tokenized_units = [lexical_tokens(unit.text) for unit in units]
    bm25 = BM25Okapi(tokenized_units)

    query_terms: list[str] = []
    for anchor in anchors:
        query_terms.extend(lexical_tokens(anchor))
    bm25_raw = (
        bm25.get_scores(query_terms).tolist() if query_terms else [0.0] * len(units)
    )
    bm25_norm = _minmax(bm25_raw, cfg.epsilon)

    tfidf = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    try:
        matrix = tfidf.fit_transform([unit.text for unit in units])
        similarity = cosine_similarity(matrix)
    except ValueError:
        similarity = [[0.0 for _ in units] for _ in units]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(units)))
    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            weight = (
                similarity[left][right]
                if isinstance(similarity, list)
                else similarity[left, right]
            )
            if weight <= 0:
                continue
            graph.add_edge(left, right, weight=float(weight))
    centrality_raw = nx.pagerank(graph, weight="weight") if len(units) > 1 else {0: 1.0}
    centrality_norm = _minmax(
        [centrality_raw.get(i, 0.0) for i in range(len(units))], cfg.epsilon
    )

    ranked_anchor_terms = [
        item[0]
        for item in sorted(anchors.items(), key=lambda item: item[1], reverse=True)
    ]
    ranked_anchor_terms = ranked_anchor_terms[
        : max(1, min(24, len(ranked_anchor_terms)))
    ]

    previous_shingles: list[set[str]] = []
    for index, unit in enumerate(units):
        tokens = lexical_tokens(unit.text)
        protected_mass = _protected_mass(unit, cfg)

        lower = unit.text.lower()
        keyphrase_density = 0.0
        for rank, phrase in enumerate(ranked_anchor_terms, start=1):
            if phrase in lower:
                keyphrase_density += 1.0 / (1.0 + rank)
        keyphrase_density = min(1.0, keyphrase_density)

        structure = _structure_score(unit, index, len(units))
        exactness = _exactness_score(unit)
        boilerplate = _boilerplate_penalty(unit, cfg)

        shingles = shingle_set(tokens, cfg.shingle_size)
        redundancy = 0.0
        for existing in previous_shingles:
            union = len(existing | shingles) or 1
            redundancy = max(redundancy, len(existing & shingles) / union)
        previous_shingles.append(shingles)

        position = _position_prior(unit, index, len(units))

        weights = cfg.raw_score_weights
        raw_score = (
            weights["protected_mass"] * protected_mass
            + weights["bm25_relevance"] * bm25_norm[index]
            + weights["textrank"] * centrality_norm[index]
            + weights["yake_density"] * keyphrase_density
            + weights["structure"] * structure
            + weights["exactness"] * exactness
            + weights["position"] * position
            + weights["redundancy_penalty"] * redundancy
            + weights["boilerplate_penalty"] * boilerplate
        )

        unit.features = {
            "protected_mass": protected_mass,
            "bm25_relevance": bm25_norm[index],
            "textrank": centrality_norm[index],
            "yake_density": keyphrase_density,
            "structure": structure,
            "exactness": exactness,
            "boilerplate": boilerplate,
            "redundancy": redundancy,
            "position": position,
        }
        unit.raw_score = float(raw_score)

    return units
