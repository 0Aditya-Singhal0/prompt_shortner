from dataclasses import dataclass, field, replace


@dataclass
class Config:
    target_reduction: float = 0.30
    keep_ratio: float = 0.70
    keep_ratio_min: float = 0.65
    keep_ratio_max: float = 0.85
    strict_mode: bool = False
    domain: str = "general"
    tokenizer_model: str = "gpt-4o-mini"

    hard_keep_protected_mass: float = 0.35
    drop_protected_mass_max: float = 0.10
    rewrite_protected_mass_max: float = 0.15
    dedupe_hard: float = 0.85
    dedupe_soft: float = 0.70
    shingle_size: int = 5
    mmr_lambda: float = 0.72
    anchor_recall_min: float = 0.90
    lexical_sim_min: float = 0.78
    boilerplate_high: float = 0.45
    structure_high: float = 0.60
    anchor_top_yake: int = 18
    anchor_top_idf: int = 14
    anchor_max_terms: int = 20
    epsilon: float = 1e-9

    output_keep_ratio_default: float = 0.82
    output_fluff_strip_threshold: float = 0.18
    output_fragment_allow_threshold: float = 0.78
    output_risk_bypass_threshold: float = 0.35
    output_protected_mass_bypass: float = 0.40
    use_output_normalizer: bool = True

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

    output_hedges: tuple[str, ...] = (
        "likely",
        "probably",
        "perhaps",
        "generally",
        "usually",
        "maybe",
        "might",
    )
    output_pleasantries: tuple[str, ...] = (
        "sure",
        "happy to help",
        "of course",
        "here's",
        "lets",
        "let's",
    )
    output_meta_phrases: tuple[str, ...] = (
        "the reason is",
        "what this means is",
        "in other words",
        "to summarize",
        "it is important to note that",
        "you should understand that",
    )
    output_risk_terms: tuple[str, ...] = (
        "delete",
        "drop",
        "irreversible",
        "cannot be undone",
        "security",
        "credential",
        "secret",
        "token",
        "production",
        "billing",
        "force",
    )

    output_fluff_patterns: tuple[str, ...] = (
        r"\b[Ss]ure[,!]?\s*",
        r"\b[Cc]ertainly[,!]?\s*",
        r"\b[Hh]appy to help[,!]?\s*",
        r"\b[Oo]f course[,!]?\s*",
        r"\b[Ii]t is important to note that\b",
        r"\b[Yy]ou should understand that\b",
        r"\b[Ii]n other words\b",
        r"\b[Tt]he reason is\b",
    )

    output_prunable_modifiers: tuple[str, ...] = (
        "very",
        "really",
        "quite",
        "extremely",
        "highly",
        "basically",
        "however",
        "therefore",
        "additionally",
    )

    def with_keep_ratio(self, keep_ratio: float) -> "Config":
        keep_ratio = min(max(keep_ratio, self.keep_ratio_min), self.keep_ratio_max)
        return replace(self, keep_ratio=keep_ratio)

    def output_keep_ratio(self) -> float:
        return min(0.90, self.keep_ratio + 0.10)

    def for_domain(self) -> "Config":
        if self.domain in {"coding", "tool_use"}:
            return replace(
                self,
                keep_ratio=max(self.keep_ratio, 0.78),
                anchor_recall_min=max(self.anchor_recall_min, 0.95),
            )
        if self.domain == "legal_like":
            return replace(
                self,
                keep_ratio=max(self.keep_ratio, 0.85),
                rewrite_protected_mass_max=min(self.rewrite_protected_mass_max, 0.05),
            )
        return self
