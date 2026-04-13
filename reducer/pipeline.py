from .anchors import extract_anchors
from .config import Config
from .dedupe import dedupe_units
from .features import compute_features
from .normalize import normalize
from .output_normalize import normalize_output
from .protect import detect_protected_spans
from .rewrite import rewrite_eligible, safe_rewrite
from .schemas import CompressionResult, Unit
from .segment import segment
from .select import select_units
from .tokenize import token_count
from .verify import verify


def _clone_unit(unit: Unit, text: str | None = None) -> Unit:
    return Unit(
        unit_id=unit.unit_id,
        text=unit.text if text is None else text,
        unit_type=unit.unit_type,
        start=unit.start,
        end=unit.end,
        token_count=unit.token_count,
        protected_spans=list(unit.protected_spans),
        hard_keep=unit.hard_keep,
        features=dict(unit.features),
        raw_score=unit.raw_score,
    )


def _assemble(units: list[Unit]) -> str:
    ordered = sorted(units, key=lambda unit: (unit.start, unit.end))
    return "\n".join(unit.text.strip() for unit in ordered if unit.text.strip())


def _ensure_min_keep_ratio(
    selected: list[Unit],
    all_units: list[Unit],
    cfg: Config,
    original_tokens: int,
    budget: int,
) -> list[Unit]:
    min_keep_tokens = int((cfg.keep_ratio_min * original_tokens) + 0.999)
    selected_tokens = sum(unit.token_count for unit in selected)
    if selected_tokens >= min_keep_tokens:
        return sorted(selected, key=lambda unit: (unit.start, unit.end))

    selected_ids = {unit.unit_id for unit in selected}
    remaining = [unit for unit in all_units if unit.unit_id not in selected_ids]
    remaining.sort(key=lambda unit: unit.raw_score, reverse=True)

    augmented = list(selected)
    for unit in remaining:
        if selected_tokens >= min_keep_tokens:
            break
        if unit.token_count < 4 or unit.raw_score < 0.5:
            continue
        if selected_tokens + unit.token_count > budget:
            continue
        augmented.append(unit)
        selected_tokens += unit.token_count
    return sorted(augmented, key=lambda unit: (unit.start, unit.end))


def _apply_safe_rewrites(
    units: list[Unit], cfg: Config
) -> tuple[list[Unit], list[str]]:
    rewritten_ids: list[str] = []
    out: list[Unit] = []
    for unit in units:
        candidate = _clone_unit(unit)
        if rewrite_eligible(candidate, cfg):
            rewritten = safe_rewrite(candidate.text, candidate.protected_spans)
            if rewritten != candidate.text:
                candidate.text = rewritten
                candidate.token_count = token_count(rewritten, cfg.tokenizer_model)
                rewritten_ids.append(candidate.unit_id)
        out.append(candidate)
    return out, rewritten_ids


def _compress_once(
    units: list[Unit], cfg: Config, budget: int, over_budget_hard_keep: bool
) -> tuple[str, list[str]]:
    if over_budget_hard_keep:
        assembled = _assemble(units)
        normalized = normalize_output(assembled, cfg, cfg.output_keep_ratio())
        return normalized, []

    rewritten_units, rewritten_ids = _apply_safe_rewrites(units, cfg)
    assembled = _assemble(rewritten_units)
    normalized = normalize_output(assembled, cfg, cfg.output_keep_ratio())
    if token_count(normalized, cfg.tokenizer_model) > budget:
        assembled = _assemble(units)
        normalized = normalize_output(assembled, cfg, cfg.output_keep_ratio())
        return normalized, []
    return normalized, rewritten_ids


def compress_prompt(text: str, cfg: Config | None = None) -> CompressionResult:
    cfg = (cfg or Config()).for_domain()
    original = normalize(text)

    spans = detect_protected_spans(original, cfg)
    units = segment(original, spans, cfg)
    anchors = extract_anchors(original, units, spans, cfg)
    units = compute_features(units, anchors, cfg)
    units = dedupe_units(units, cfg)

    original_tokens = token_count(original, cfg.tokenizer_model)
    budget = int((cfg.keep_ratio * original_tokens) + 0.999)

    selected, over_budget_hard_keep = select_units(units, original_tokens, cfg)
    selected = _ensure_min_keep_ratio(selected, units, cfg, original_tokens, budget)
    compressed, rewritten_ids = _compress_once(
        selected, cfg, budget, over_budget_hard_keep
    )

    verification = verify(
        original, compressed, spans, anchors, cfg, token_budget=budget
    )

    if not verification["passed"]:
        extracted_only = _assemble(selected)
        extracted_only = normalize_output(extracted_only, cfg, cfg.output_keep_ratio())
        verification_no_rewrite = verify(
            original,
            extracted_only,
            spans,
            anchors,
            cfg,
            token_budget=budget,
        )
        if verification_no_rewrite["passed"]:
            compressed = extracted_only
            verification = verification_no_rewrite
            rewritten_ids = []
        else:
            retry_cfg = cfg.with_keep_ratio(cfg.keep_ratio + 0.05)
            retry_budget = int((retry_cfg.keep_ratio * original_tokens) + 0.999)
            retry_selected, retry_over_budget = select_units(
                units, original_tokens, retry_cfg
            )
            retry_selected = _ensure_min_keep_ratio(
                retry_selected,
                units,
                retry_cfg,
                original_tokens,
                retry_budget,
            )
            retry_compressed, retry_rewrites = _compress_once(
                retry_selected,
                retry_cfg,
                retry_budget,
                retry_over_budget,
            )
            retry_verification = verify(
                original,
                retry_compressed,
                spans,
                anchors,
                retry_cfg,
                token_budget=retry_budget,
            )
            if retry_verification["passed"]:
                selected = retry_selected
                compressed = retry_compressed
                verification = retry_verification
                rewritten_ids = retry_rewrites
                cfg = retry_cfg
                budget = retry_budget
            else:
                compressed = original
                rewritten_ids = []
                selected = units
                verification = verify(
                    original,
                    compressed,
                    spans,
                    anchors,
                    cfg,
                    token_budget=original_tokens,
                )

    compressed_tokens = token_count(compressed, cfg.tokenizer_model)
    kept_ids = [unit.unit_id for unit in selected]
    dropped_ids = [unit.unit_id for unit in units if unit.unit_id not in kept_ids]
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
        verification=verification,
    )
