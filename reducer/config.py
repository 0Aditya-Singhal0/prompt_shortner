from dataclasses import dataclass, field


@dataclass
class Config:
    keep_ratio: float = 0.70
    keep_ratio_min: float = 0.65
    keep_ratio_max: float = 0.85
    strict_mode: bool = False
    domain: str = "general"

    hard_keep_protected_mass: float = 0.35
    drop_protected_mass_max: float = 0.10
    rewrite_protected_mass_max: float = 0.15
    dedupe_hard: float = 0.85
    dedupe_soft: float = 0.70
    mmr_lambda: float = 0.72
    anchor_recall_min: float = 0.90
    lexical_sim_min: float = 0.78
    boilerplate_high: float = 0.45
    structure_high: float = 0.60

    raw_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "protected_mass": 3.0,
            "bm25_relevance": 1.8,
            "textrank": 1.2,
            "yake_density": 1.0,
            "structure": 1.1,
            "exactness": 0.9,
            "position": 0.3,
            "redundancy_penalty": -1.5,
            "boilerplate_penalty": -1.2,
        }
    )

    negation_lexicon: tuple[str, ...] = (
        "must",
        "must not",
        "do not",
        "never",
        "only",
        "exactly",
        "at least",
        "at most",
        "no more than",
        "without",
        "cannot",
    )

    boilerplate_terms: tuple[str, ...] = (
        "please",
        "kindly",
        "basically",
        "actually",
        "just",
        "i want you to",
        "can you",
        "it would be great if",
        "in a detailed manner",
    )

    def with_keep_ratio(self, keep_ratio: float) -> "Config":
        keep_ratio = min(max(keep_ratio, self.keep_ratio_min), self.keep_ratio_max)
        clone = Config(**{**self.__dict__, "keep_ratio": keep_ratio})
        return clone
