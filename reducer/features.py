import math

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .schemas import Unit


def _norm(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-9:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo + 1e-9) for v in vals]


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def _structure_score(text: str, idx: int, n: int, cfg: Config) -> float:
    s = 0.0
    t = text.lower()
    if text.strip().startswith(tuple(f"{k}." for k in range(1, 10))):
        s += 0.30
    if text.strip().startswith(("-", "*")):
        s += 0.20
    if ":" in text:
        s += 0.20
    if _contains_any(t, ("must", "never", "only", "exactly")):
        s += 0.35
    if idx == n - 1 and "?" in text:
        s += 0.30
    if _contains_any(t, ("output", "format", "json", "table", "markdown", "return")):
        s += 0.25
    return min(1.0, s)


def _boilerplate(text: str, cfg: Config) -> float:
    t = text.lower()
    words = t.split()
    if not words:
        return 0.0
    density = sum(t.count(p) for p in cfg.boilerplate_terms) / max(len(words), 1)
    stop_ratio = sum(
        w in {"the", "a", "an", "is", "are", "to", "of", "in", "that", "for"}
        for w in words
    ) / len(words)
    hedge_ratio = sum(
        w in {"maybe", "perhaps", "possibly", "somewhat"} for w in words
    ) / len(words)
    return min(1.0, 0.5 * density + 0.3 * stop_ratio + 0.2 * hedge_ratio)


def compute_features(
    units: list[Unit], anchors: dict[str, float], cfg: Config
) -> list[Unit]:
    if not units:
        return units

    tokenized = [u.text.lower().split() for u in units]
    bm25 = BM25Okapi(tokenized)
    query = [a for a in anchors.keys() if " " not in a]
    r = bm25.get_scores(query).tolist() if query else [0.0] * len(units)
    rhat = _norm(r)

    vec = TfidfVectorizer(lowercase=True)
    X = vec.fit_transform([u.text for u in units])
    sim = cosine_similarity(X)
    g = nx.Graph()
    g.add_nodes_from(range(len(units)))
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if sim[i, j] > 0:
                g.add_edge(i, j, weight=float(sim[i, j]))
    pr = nx.pagerank(g, weight="weight") if len(units) > 1 else {0: 1.0}
    chat = _norm([pr.get(i, 0.0) for i in range(len(units))])

    prev_shingles: list[set[str]] = []
    for i, u in enumerate(units):
        protected_tokens = sum(
            max(1, (s.end - s.start) // 4) for s in u.protected_spans
        )
        p_i = min(1.0, protected_tokens / max(u.token_count, 1))

        lower = u.text.lower()
        k_i = 0.0
        for rank, key in enumerate(anchors.keys(), start=1):
            if key in lower:
                k_i += 1.0 / (1.0 + rank)
        k_i = min(1.0, k_i)

        s_i = _structure_score(u.text, i, len(units), cfg)
        e_i = min(
            1.0,
            (
                sum(s.label == "IDENTIFIER" for s in u.protected_spans)
                + sum(
                    s.label in {"NUMBER", "DATE", "TIME", "SEMVER"}
                    for s in u.protected_spans
                )
                + sum(s.label == "QUOTE" for s in u.protected_spans)
            )
            / 3.0,
        )
        b_i = _boilerplate(u.text, cfg)

        words = u.text.lower().split()
        shingles = set(
            " ".join(words[k : k + 5]) for k in range(max(0, len(words) - 4))
        )
        d_i = 0.0
        for old in prev_shingles:
            union = len(old | shingles) or 1
            d_i = max(d_i, len(old & shingles) / union)
        prev_shingles.append(shingles)

        l_i = 0.3 if i == len(units) - 1 and "?" in u.text else (0.2 if i == 0 else 0.0)

        w = cfg.raw_score_weights
        raw = (
            w["protected_mass"] * p_i
            + w["bm25_relevance"] * rhat[i]
            + w["textrank"] * chat[i]
            + w["yake_density"] * k_i
            + w["structure"] * s_i
            + w["exactness"] * e_i
            + w["position"] * l_i
            + w["redundancy_penalty"] * d_i
            + w["boilerplate_penalty"] * b_i
        )

        u.features = {
            "protected_mass": p_i,
            "bm25_relevance": rhat[i],
            "textrank": chat[i],
            "yake_density": k_i,
            "structure": s_i,
            "exactness": e_i,
            "boilerplate": b_i,
            "redundancy": d_i,
            "position": l_i,
        }
        u.raw_score = raw
    return units
