from reducer.config import Config
from reducer.protect import detect_protected_spans
from reducer.segment import segment


def test_keeps_code_block_atomic() -> None:
    text = "One.\n```python\nprint('x')\n```\nTwo."
    spans = detect_protected_spans(text, Config())
    units = segment(text, spans)
    assert any(u.unit_type == "CODE_BLOCK" for u in units)
    code_units = [u for u in units if u.unit_type == "CODE_BLOCK"]
    assert len(code_units) == 1
