from .anchors import extract_anchors
from .config import Config
from .dedupe import dedupe_units
from .features import compute_features
from .normalize import normalize
from .protect import detect_protected_spans
from .rewrite import rewrite_eligible, safe_rewrite
from .schemas import CompressionResult
from .segment import segment
from .select import select_units
from .tokenize import token_count
from .verify import verify


def _assemble(units):
    units = sorted(units, key=lambda u: u.start)
    return "\n".join(u.text.strip() for u in units if u.text.strip())


def compress_prompt(text: str, cfg: Config | None = None) -> CompressionResult:
    cfg = cfg or Config()
    original = normalize(text)
    spans = detect_protected_spans(original, cfg)
    units = segment(original, spans)
    anchors = extract_anchors(original, units, spans)
    units = compute_features(units, anchors, cfg)
    units = dedupe_units(units, cfg)

    original_tokens = token_count(original)
    selected = select_units(units, original_tokens, cfg)

    rewritten_ids: list[str] = []
    rewritten_units = []
    for u in selected:
        if rewrite_eligible(u, cfg):
            new_text = safe_rewrite(u.text)
            if new_text != u.text:
                rewritten_ids.append(u.unit_id)
                u.text = new_text
        rewritten_units.append(u)

    compressed = _assemble(rewritten_units)
    ver = verify(original, compressed, spans, anchors, cfg)

    if not ver["passed"]:
        no_rewrite = _assemble(sorted(selected, key=lambda u: u.start))
        ver2 = verify(original, no_rewrite, spans, anchors, cfg)
        if ver2["passed"]:
            compressed = no_rewrite
            ver = ver2
        else:
            cfg2 = cfg.with_keep_ratio(cfg.keep_ratio + 0.05)
            selected2 = select_units(units, original_tokens, cfg2)
            compressed2 = _assemble(selected2)
            ver3 = verify(original, compressed2, spans, anchors, cfg2)
            if ver3["passed"]:
                compressed = compressed2
                ver = ver3
                selected = selected2
                rewritten_ids = []
            else:
                compressed = original
                ver = verify(original, compressed, spans, anchors, cfg)
                selected = units
                rewritten_ids = []

    compressed_tokens = token_count(compressed)
    kept_ids = [u.unit_id for u in selected]
    dropped_ids = [u.unit_id for u in units if u.unit_id not in kept_ids]
    reduction_ratio = (
        0.0 if original_tokens == 0 else 1.0 - (compressed_tokens / original_tokens)
    )

    return CompressionResult(
        original_text=original,
        compressed_text=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        reduction_ratio=round(reduction_ratio, 4),
        protected_spans=spans,
        kept_units=kept_ids,
        dropped_units=dropped_ids,
        rewritten_units=rewritten_ids,
        verification=ver,
    )
