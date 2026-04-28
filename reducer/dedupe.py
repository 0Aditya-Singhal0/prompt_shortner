from datasketch import MinHash

from .config import Config
from .schemas import Unit
from .tokenize import lexical_tokens, shingle_set


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right) or 1
    return len(left & right) / union


def _minhash_signature(shingles: set[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in sorted(shingles):
        mh.update(shingle.encode("utf-8"))
    return mh


def dedupe_units(units: list[Unit], cfg: Config) -> list[Unit]:
    if not units:
        return units

    ranked = sorted(units, key=lambda unit: unit.raw_score, reverse=True)
    kept: list[Unit] = []
    kept_shingles: list[set[str]] = []
    kept_minhashes: list[MinHash] = []

    for unit in ranked:
        shingles = shingle_set(lexical_tokens(unit.text), cfg.shingle_size)
        minhash = _minhash_signature(shingles, cfg.minhash_num_perm)
        hard_duplicate = False
        soft_duplicate = False

        for index, existing in enumerate(kept_shingles):
            if cfg.use_minhash_dedupe:
                minhash_sim = minhash.jaccard(kept_minhashes[index])
                if minhash_sim + cfg.minhash_filter_margin < cfg.dedupe_soft:
                    continue

            resemblance = _jaccard(shingles, existing)
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
        kept_minhashes.append(minhash)

    kept.sort(key=lambda unit: (unit.start, unit.end))
    return kept
