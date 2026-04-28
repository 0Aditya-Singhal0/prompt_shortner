from reducer.config import Config
from reducer.features import compute_features
from reducer.schemas import Unit


def _make_unit(unit_id: str, text: str) -> Unit:
    return Unit(
        unit_id=unit_id,
        text=text,
        unit_type="SENTENCE",
        start=0,
        end=len(text),
        token_count=max(1, len(text.split())),
        protected_spans=[],
    )


def test_compute_features_short_inputs_skip_heavy_models(monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("heavy feature model should not run")

    monkeypatch.setattr("reducer.features.BM25Okapi", _boom)
    monkeypatch.setattr("reducer.features.TfidfVectorizer", _boom)

    units = [_make_unit("u0", "Fix cache."), _make_unit("u1", "Return JSON.")]
    out = compute_features(units, {"cache": 1.0}, Config())

    assert len(out) == 2
    assert all(unit.features["bm25_relevance"] == 0.0 for unit in out)
    assert all(unit.features["textrank"] == 0.0 for unit in out)
