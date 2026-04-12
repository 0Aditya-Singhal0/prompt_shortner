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
