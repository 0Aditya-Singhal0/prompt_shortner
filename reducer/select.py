import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .schemas import Unit


def _contains_imperative(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:return|provide|generate|create|build|explain|fix|implement|output)\b",
            text,
            re.IGNORECASE,
        )
    )


def hard_keep_rule(unit: Unit, cfg: Config) -> bool:
    features = unit.features
    labels = {span.label for span in unit.protected_spans}
    lower = unit.text.lower()

    if unit.unit_type == "CODE_BLOCK":
        return True
    if unit.unit_type == "KEY_VALUE_LINE" and ":" in unit.text:
        return True
    if len(unit.protected_spans) >= 2:
        return True
    if "NEGATION" in labels:
        return True
    if features.get("protected_mass", 0.0) >= cfg.hard_keep_protected_mass:
        return True
    if features.get("structure", 0.0) >= cfg.structure_high:
        return True
    if "OUTPUT_FORMAT" in labels:
        return True
    if "NUMBER" in labels and "NEGATION" in labels:
        return True
    if re.match(r"^\d+[.)]\s+", unit.text.strip()):
        return True
    if "QUOTE" in labels and _contains_imperative(unit.text):
        return True
    if any(
        token in lower
        for token in ("output format", "json", "markdown", "table", "schema")
    ):
        return True
    if unit.text.strip().endswith("?"):
        return True
    return False


def drop_candidate_rule(unit: Unit, cfg: Config) -> bool:
    features = unit.features
    return (
        features.get("protected_mass", 0.0) < cfg.drop_protected_mass_max
        and features.get("structure", 0.0) < 0.25
        and features.get("exactness", 0.0) < 0.20
        and features.get("boilerplate", 0.0) > cfg.boilerplate_high
        and (features.get("redundancy", 0.0) > 0.80 or unit.raw_score < 0.25)
    )


def _similarity_matrix(units: list[Unit]):
    if not units:
        return None, {}
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    try:
        matrix = vectorizer.fit_transform([unit.text for unit in units])
    except ValueError:
        return None, {unit.unit_id: index for index, unit in enumerate(units)}
    similarity = cosine_similarity(matrix)
    mapping = {unit.unit_id: index for index, unit in enumerate(units)}
    return similarity, mapping


def select_units(
    units: list[Unit], original_tokens: int, cfg: Config
) -> tuple[list[Unit], bool]:
    if not units:
        return [], False

    for unit in units:
        unit.hard_keep = hard_keep_rule(unit, cfg)

    budget = int((cfg.keep_ratio * original_tokens) + 0.999)
    selected = [unit for unit in units if unit.hard_keep]
    budget_used = sum(unit.token_count for unit in selected)
    if budget_used > budget:
        return sorted(selected, key=lambda unit: (unit.start, unit.end)), True

    candidates = [
        unit
        for unit in units
        if unit not in selected and not drop_candidate_rule(unit, cfg)
    ]
    similarity, mapping = _similarity_matrix(units)

    while candidates and budget_used < budget:
        best_unit = None
        best_mmr = float("-inf")

        for unit in candidates:
            if budget_used + unit.token_count > budget:
                continue

            if not selected or similarity is None:
                redundancy = 0.0
            else:
                redundancy = max(
                    similarity[mapping[unit.unit_id], mapping[chosen.unit_id]]
                    for chosen in selected
                )

            mmr = cfg.mmr_lambda * unit.raw_score - (1.0 - cfg.mmr_lambda) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best_unit = unit

        if best_unit is None or best_mmr <= 0:
            break

        selected.append(best_unit)
        candidates.remove(best_unit)
        budget_used += best_unit.token_count

    selected.sort(key=lambda unit: (unit.start, unit.end))
    return selected, False
