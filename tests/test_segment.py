from reducer.config import Config
from reducer.protect import detect_protected_spans
from reducer.segment import segment


def test_keeps_code_block_atomic() -> None:
    text = "One.\n```python\nprint('x')\n```\nTwo."
    cfg = Config()
    spans = detect_protected_spans(text, cfg)
    units = segment(text, spans, cfg)
    assert any(u.unit_type == "CODE_BLOCK" for u in units)
    code_units = [u for u in units if u.unit_type == "CODE_BLOCK"]
    assert len(code_units) == 1


def test_splits_lists_and_key_value_lines() -> None:
    text = "- first item\n- second item\nmode: strict\n"
    cfg = Config()
    spans = detect_protected_spans(text, cfg)
    units = segment(text, spans, cfg)
    unit_types = [unit.unit_type for unit in units]
    assert unit_types.count("LIST_ITEM") == 2
    assert "KEY_VALUE_LINE" in unit_types


def test_inline_fence_start_does_not_create_fake_sentence_unit() -> None:
    text = (
        "Can you examine the following code and identify any issues? "
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```"
    )
    cfg = Config()
    spans = detect_protected_spans(text, cfg)
    units = segment(text, spans, cfg)
    assert sum(unit.unit_type == "CODE_BLOCK" for unit in units) == 1
    assert not any(
        unit.unit_type != "CODE_BLOCK" and unit.text.strip().startswith("```")
        for unit in units
    )
