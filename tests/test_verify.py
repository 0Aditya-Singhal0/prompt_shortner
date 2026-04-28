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


def test_verify_fails_when_repeated_flag_occurrence_is_lost() -> None:
    original = "Run --dry-run first, then run --dry-run again before apply."
    compressed = "Run --dry-run before apply."
    cfg = Config()
    spans = detect_protected_spans(original, cfg)
    out = verify(original, compressed, spans, {"dry-run": 1.0}, cfg)
    assert out["protected_coverage"] < 1.0
    assert out["passed"] is False


def test_verify_fails_when_repeated_issue_id_occurrence_is_lost() -> None:
    original = "Link #123 to #123 in the release notes."
    compressed = "Link #123 in the release notes."
    cfg = Config()
    spans = detect_protected_spans(original, cfg)
    out = verify(original, compressed, spans, {"123": 1.0}, cfg)
    assert out["protected_coverage"] < 1.0
    assert out["passed"] is False


def test_verify_fails_when_repeated_file_path_occurrence_is_lost() -> None:
    original = "Check /tmp/data.csv and archive /tmp/data.csv before upload."
    compressed = "Check /tmp/data.csv before upload."
    cfg = Config()
    spans = detect_protected_spans(original, cfg)
    out = verify(original, compressed, spans, {"data.csv": 1.0}, cfg)
    assert out["numeric_coverage"] == 0.0
    assert out["passed"] is False
