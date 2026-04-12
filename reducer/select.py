from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .schemas import Unit


def hard_keep_rule(unit: Unit, cfg: Config) -> bool:
    f = unit.features
    text = unit.text.lower()
    labels = {s.label for s in unit.protected_spans}
    return any(
        [
            unit.unit_type == "CODE_BLOCK",
            len(unit.protected_spans) >= 2,
            "NEGATION" in labels,
            f.get("protected_mass", 0.0) >= cfg.hard_keep_protected_mass,
            f.get("structure", 0.0) >= cfg.structure_high,
            any(w in text for w in ("output format", "json", "markdown", "table")),
            ("NUMBER" in labels and "NEGATION" in labels),
            text.endswith("?"),
        ]
    )


def select_units(units: list[Unit], original_tokens: int, cfg: Config) -> list[Unit]:
    if not units:
        return []
    for u in units:
        u.hard_keep = hard_keep_rule(u, cfg)

    budget = int((cfg.keep_ratio * original_tokens) + 0.999)
    selected = [u for u in units if u.hard_keep]
    used = sum(u.token_count for u in selected)
    if used > budget:
        return selected

    rest = [u for u in units if u not in selected]
    corpus = [u.text for u in units]
    vec = TfidfVectorizer(lowercase=True)
    X = vec.fit_transform(corpus)
    sim = cosine_similarity(X)
    idx = {u.unit_id: i for i, u in enumerate(units)}

    while rest and used < budget:
        best = None
        best_score = float("-inf")
        for u in rest:
            if u.token_count + used > budget:
                continue
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(sim[idx[u.unit_id], idx[s.unit_id]] for s in selected)
            mmr = cfg.mmr_lambda * u.raw_score - (1 - cfg.mmr_lambda) * redundancy
            if mmr > best_score:
                best = u
                best_score = mmr
        if best is None or best_score <= 0:
            break
        selected.append(best)
        rest.remove(best)
        used += best.token_count
    return selected
