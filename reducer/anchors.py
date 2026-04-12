from collections import Counter

import yake
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import Span, Unit


def _tokenize_simple(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.strip()]


def extract_anchors(
    text: str, units: list[Unit], spans: list[Span]
) -> dict[str, float]:
    anchors: dict[str, float] = {}
    for s in spans:
        t = s.text.strip().lower()
        if t:
            bonus = (
                1.5 if s.label in {"INLINE_CODE", "CODE_BLOCK", "IDENTIFIER"} else 1.2
            )
            anchors[t] = max(anchors.get(t, 0.0), bonus)

    kw = yake.KeywordExtractor(lan="en", n=3, top=20).extract_keywords(text)
    for phrase, score in kw:
        p = str(phrase).lower().strip()
        if not p:
            continue
        salience = 1.0 / (1.0 + float(score))
        anchors[p] = max(anchors.get(p, 0.0), 0.8 * salience)

    docs = [u.text for u in units if u.text.strip()]
    if docs:
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), max_features=5000)
        X = vec.fit_transform(docs)
        names = vec.get_feature_names_out()
        idf = vec.idf_
        for term, value in zip(names, idf):
            anchors[term] = max(
                anchors.get(term, 0.0), float(value) / (max(idf) + 1e-9)
            )

    freq = Counter(_tokenize_simple(text))
    for a in list(anchors):
        anchors[a] += min(0.6, 0.15 * freq.get(a, 0))
    return anchors
