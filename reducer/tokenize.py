import re

import tiktoken


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_LEX_RE = re.compile(r"[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)?")


def approx_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def lexical_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _LEX_RE.finditer(text)]


def shingle_set(tokens: list[str], k: int = 5) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def token_count(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
