from reducer.config import Config
from reducer.pipeline import compress_prompt, run_inference_pipeline


def test_pipeline_reduces_redundant_prompt_and_preserves_flag() -> None:
    text = (
        "Can you please explain caching and also explain caching and also explain caching in web apps. "
        "Do not remove `--flag`. Return JSON."
    )
    result = compress_prompt(text, Config())
    assert result.compressed_tokens <= result.original_tokens
    assert "`--flag`" in result.compressed_text
    assert result.verification["passed"] is True


def test_pipeline_preserves_fenced_code_exactly() -> None:
    text = (
        "Can you examine the following code and identify any potential issues? "
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```"
    )
    result = compress_prompt(text, Config())
    assert result.compressed_text.count("```python") == 1
    assert "def add(a, b):" in result.compressed_text
    assert "return a + b" in result.compressed_text
    assert result.verification["passed"] is True


def test_full_pipeline_runs_external_model_then_normalizer() -> None:
    seen_inputs: list[str] = []

    def model_fn(compressed_input: str) -> str:
        seen_inputs.append(compressed_input)
        return "Sure, happy to help. Return JSON."

    result = run_inference_pipeline(
        "Please explain the migration plan and return JSON.", model_fn, Config()
    )
    assert seen_inputs and seen_inputs[0] == result.compressed_input
    assert result.raw_output == "Sure, happy to help. Return JSON."
    assert "Sure" not in result.normalized_output
