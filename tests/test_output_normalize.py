from reducer.config import Config
import reducer.output_normalize as output_normalize
from reducer.output_normalize import normalize_output


def test_output_normalizer_preserves_code_and_urls() -> None:
    text = (
        "Sure, I am happy to help. The reason is this should be quick.\n"
        "```bash\n"
        "curl https://example.com/api --header 'X-Test: 1'\n"
        "```\n"
    )
    out = normalize_output(text, Config())
    assert "curl https://example.com/api --header 'X-Test: 1'" in out
    assert "Sure" not in out


def test_output_normalizer_keeps_instructional_parentheticals() -> None:
    text = "Return retries (at most 3) and timeout (do not exceed 30s)."
    out = normalize_output(text, Config())
    assert "(at most 3)" in out
    assert "(do not exceed 30s)" in out


def test_output_normalizer_prunes_low_signal_parentheticals() -> None:
    text = "Use cached values (for example) when available."
    out = normalize_output(text, Config())
    assert "(for example)" not in out


def test_output_normalizer_prunes_with_pos_tags(monkeypatch) -> None:
    monkeypatch.setattr(
        output_normalize,
        "_pos_tag_tokens",
        lambda _text: [
            ("This", "DT"),
            ("is", "VBZ"),
            ("very", "RB"),
            ("quick", "JJ"),
            (".", "."),
        ],
    )
    out = normalize_output("This is very quick.", Config())
    assert "very" not in out.lower()
    assert "quick" not in out.lower()


def test_output_normalizer_prunes_leading_subordinate_clause() -> None:
    text = "Because this is redundant, return JSON with id."
    out = normalize_output(text, Config())
    assert out.lower().startswith("return json")
