import re

from .config import Config
from .schemas import Unit


def rewrite_eligible(unit: Unit, cfg: Config) -> bool:
    if unit.hard_keep:
        return False
    if unit.features.get("protected_mass", 1.0) >= cfg.rewrite_protected_mass_max:
        return False
    labels = {s.label for s in unit.protected_spans}
    blocked = {
        "CODE_BLOCK",
        "FILE_PATH",
        "SEMVER",
        "QUOTE",
        "NUMBER",
        "NEGATION",
        "URL",
        "INLINE_CODE",
    }
    if labels & blocked:
        return False
    return unit.token_count >= 12


def safe_rewrite(text: str) -> str:
    rules = [
        (r"\b[Cc]an you please\b", ""),
        (r"\b[Ii] want you to\b", ""),
        (r"\b[Bb]asically\b", ""),
        (r"\b[Ii]t would be great if you could\b", ""),
        (r"\b[Ii] need you to provide\b", "Provide"),
        (r"\bvery\b|\breally\b|\bquite\b|\bhighly\b|\bextremely\b", ""),
        (r"\band also\b", ", "),
    ]
    out = text
    for pat, rep in rules:
        out = re.sub(pat, rep, out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,;")
    if out and out[-1] not in ".?!":
        out += "."
    return out or text
