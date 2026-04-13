from reducer.config import Config
from reducer.protect import detect_protected_spans


def test_detects_core_spans() -> None:
    text = "Use `func()` in /tmp/app.py with --flag and v1.2.3 on 2026-01-01."
    spans = detect_protected_spans(text, Config())
    labels = {s.label for s in spans}
    assert "INLINE_CODE" in labels
    assert "FILE_PATH" in labels
    assert "CLI_FLAG" in labels
    assert "SEMVER" in labels
    assert "DATE" in labels


def test_does_not_promote_sentence_words_to_identifier() -> None:
    text = "Can you please explain this quickly? It is urgent."
    spans = detect_protected_spans(text, Config())
    values = {span.text for span in spans if span.label == "IDENTIFIER"}
    assert "Can" not in values
    assert "It" not in values
