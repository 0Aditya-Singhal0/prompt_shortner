import re

import tiktoken


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def token_count(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
