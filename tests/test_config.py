from reducer.config import Config


def test_target_reduction_drives_keep_ratio_when_keep_ratio_is_default() -> None:
    cfg = Config(target_reduction=0.20)
    assert cfg.keep_ratio == 0.80
    assert cfg.target_reduction == 0.20


def test_keep_ratio_remains_canonical_when_both_are_set() -> None:
    cfg = Config(target_reduction=0.10, keep_ratio=0.75)
    assert cfg.keep_ratio == 0.75
    assert cfg.target_reduction == 0.25


def test_data_domain_applies_conservative_defaults() -> None:
    cfg = Config(
        domain="data",
        keep_ratio=0.70,
        anchor_recall_min=0.90,
        rewrite_protected_mass_max=0.15,
    )
    tuned = cfg.for_domain()
    assert tuned.keep_ratio >= 0.80
    assert tuned.anchor_recall_min >= 0.93
    assert tuned.rewrite_protected_mass_max <= 0.10
