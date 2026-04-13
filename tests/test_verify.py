from reducer.config import Config
from reducer.protect import detect_protected_spans
from reducer.verify import verify


def test_verify_fails_when_numeric_lost() -> None:
    original = "Do not exceed 30 ms. Return JSON."
    compressed = "Return JSON."
    cfg = Config()
    spans = detect_protected_spans(original, cfg)
    out = verify(original, compressed, spans, {"json": 1.0}, cfg)
    assert out["passed"] is False


def test_verify_requires_budget_when_given() -> None:
    original = "Return JSON with fields id and status."
    compressed = original
    cfg = Config(keep_ratio=0.50)
    spans = detect_protected_spans(original, cfg)
    out = verify(
        original, compressed, spans, {"json": 1.0, "status": 1.0}, cfg, token_budget=1
    )
    assert out["budget_ok"] is False
    assert out["passed"] is False


def test_verify_detects_code_block_modification() -> None:
    original = "Question? ```python\nprint('x')\n```"
    compressed = "Question? ```python\nprint('y')\n```"
    cfg = Config()
    spans = detect_protected_spans(original, cfg)
    out = verify(original, compressed, spans, {"print": 1.0}, cfg)
    assert out["structural_preserved"] is False
    assert out["passed"] is False
