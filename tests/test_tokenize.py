import reducer.tokenize as tokenize


class _DummyEncoding:
    def encode(self, text: str) -> list[int]:
        return [1 for _ in text.split()]


def test_token_count_caches_model_encoding(monkeypatch) -> None:
    calls: list[str] = []

    def fake_encoding_for_model(model: str):
        calls.append(model)
        return _DummyEncoding()

    monkeypatch.setattr(tokenize.tiktoken, "encoding_for_model", fake_encoding_for_model)
    monkeypatch.setattr(tokenize.tiktoken, "get_encoding", lambda _name: _DummyEncoding())
    tokenize._encoding_for_model.cache_clear()

    tokenize.token_count("alpha beta", model="custom-model")
    tokenize.token_count("gamma", model="custom-model")

    assert calls == ["custom-model"]
