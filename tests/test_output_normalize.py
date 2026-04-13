from reducer.config import Config
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
