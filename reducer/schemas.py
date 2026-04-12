from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str


@dataclass
class Unit:
    unit_id: str
    text: str
    unit_type: str
    start: int
    end: int
    token_count: int
    protected_spans: list[Span]
    hard_keep: bool = False
    features: dict[str, float] = field(default_factory=dict)
    raw_score: float = 0.0


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    reduction_ratio: float
    protected_spans: list[Span]
    kept_units: list[str]
    dropped_units: list[str]
    rewritten_units: list[str]
    verification: dict[str, Any]
