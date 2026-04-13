from .config import Config
from .schemas import Unit
from .tokenize import lexical_tokens, shingle_set


def dedupe_units(units: list[Unit], cfg: Config) -> list[Unit]:
    if not units:
        return units

    ranked = sorted(units, key=lambda unit: unit.raw_score, reverse=True)
    kept: list[Unit] = []
    kept_shingles: list[set[str]] = []

    for unit in ranked:
        shingles = shingle_set(lexical_tokens(unit.text), cfg.shingle_size)
        hard_duplicate = False
        soft_duplicate = False

        for existing in kept_shingles:
            union = len(shingles | existing) or 1
            resemblance = len(shingles & existing) / union
            if resemblance >= cfg.dedupe_hard:
                hard_duplicate = True
                break
            if resemblance >= cfg.dedupe_soft:
                soft_duplicate = True

        if hard_duplicate:
            continue

        if soft_duplicate:
            unit.features["soft_duplicate"] = 1.0
        kept.append(unit)
        kept_shingles.append(shingles)

    kept.sort(key=lambda unit: (unit.start, unit.end))
    return kept
