from reducer.config import Config
from reducer.dedupe import dedupe_units
from reducer.schemas import Unit


def _unit(unit_id: str, text: str, score: float) -> Unit:
    return Unit(
        unit_id=unit_id,
        text=text,
        unit_type="SENTENCE",
        start=0,
        end=len(text),
        token_count=max(1, len(text.split())),
        protected_spans=[],
        raw_score=score,
    )


def test_dedupe_removes_near_duplicate_with_minhash_enabled() -> None:
    cfg = Config(use_minhash_dedupe=True)
    units = [
        _unit("u0", "Return JSON with status and id fields.", 2.0),
        _unit("u1", "Return JSON with status and id fields.", 1.8),
    ]
    out = dedupe_units(units, cfg)
    assert len(out) == 1


def test_dedupe_marks_soft_duplicate_feature() -> None:
    cfg = Config(use_minhash_dedupe=True, dedupe_hard=0.95, dedupe_soft=0.2)
    units = [
        _unit("u0", "Return JSON with id status code and retry policy now.", 2.0),
        _unit("u1", "Return JSON with id status code and retry limits now.", 1.9),
    ]
    out = dedupe_units(units, cfg)
    assert len(out) == 2
    assert any(unit.features.get("soft_duplicate") == 1.0 for unit in out)
