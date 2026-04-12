from .config import Config
from .schemas import Unit


def dedupe_units(units: list[Unit], cfg: Config) -> list[Unit]:
    kept: list[Unit] = []
    for u in units:
        words = u.text.lower().split()
        sh = set(" ".join(words[i : i + 5]) for i in range(max(0, len(words) - 4)))
        duplicate = False
        for k in kept:
            kw = k.text.lower().split()
            kh = set(" ".join(kw[i : i + 5]) for i in range(max(0, len(kw) - 4)))
            union = len(sh | kh) or 1
            jac = len(sh & kh) / union
            if jac >= cfg.dedupe_hard and u.raw_score <= k.raw_score:
                duplicate = True
                break
        if not duplicate:
            kept.append(u)
    return kept
