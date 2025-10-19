# lexical.py

import heapq
import pickle
import re

from pathlib import Path
from typing import Any, Dict, List, Optional

from nltk.stem.snowball import SnowballStemmer
from rapidfuzz.fuzz import token_set_ratio

# simple unicode letters; numbers are ignored by design
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")

_VOCAB_PATH_DEFAULT = "artifacts/vocab.pkl"
_vocab: Optional[Dict[str, Any]] = None

APPLY_STOPWORDS = True  # set to False if the union is too aggressive

_stemmer = SnowballStemmer("english")

STEM_NS = "§"  # namespace prefix used when writing stems into the bloom

# --- Compatibility: TinyBloom class for old pickles built via build_vocab.py ---
class TinyBloom:
    """Simple Bloom filter used in vocab artefacts."""
    def __init__(self, m_bits: int = 4_000_000, k: int = 7):
        self.m = m_bits
        self.k = k
        self.bits = bytearray(m_bits // 8 + 1)

    def _hashes(self, s: str):
        h1 = hash(s)
        h2 = hash("§" + s)
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, s: str):
        for h in self._hashes(s):
            self.bits[h // 8] |= (1 << (h % 8))

    def __contains__(self, s: str) -> bool:
        for h in self._hashes(s):
            if not (self.bits[h // 8] & (1 << (h % 8))):
                return False
        return True


class _CompatUnpickler(pickle.Unpickler):
    """Map old module paths to current classes for legacy pickles."""
    _ALIAS = {("__main__", "TinyBloom"): TinyBloom}

    def find_class(self, module, name):
        alias = self._ALIAS.get((module, name))
        if alias:
            return alias
        return super().find_class(module, name)


def _load_stop_union() -> set[str]:
    p = Path("artifacts/stopwords_eu.pkl")
    if not p.exists():
        return set()
    with p.open("rb") as f:
        bundle = pickle.load(f)
    # keep negations; they matter semantically
    sw = set(bundle.get("all_union", set()))
    for w in ("no", "not", "nor"):
        sw.discard(w)
    return sw

_STOP_UNION = _load_stop_union()


def norm_tokens_nostop(text: str) -> List[str]:
    toks = [w.lower() for w in _WORD.findall(text)]
    toks = [t for t in toks if len(t) > 2]
    if APPLY_STOPWORDS and _STOP_UNION:
        toks = [t for t in toks if t not in _STOP_UNION]
    return toks


def explain_matches(query: str, vocab_path: str = _VOCAB_PATH_DEFAULT):
    vocab = _ensure_vocab(vocab_path)
    bloom = vocab.get("bloom")
    rows = []
    for t in norm_tokens_nostop(query):
        rows.append((t, bool(t in bloom), bool((STEM_NS + _stem(t)) in bloom)))
    return rows


def _ensure_vocab(path: str = _VOCAB_PATH_DEFAULT) -> Dict[str, Any]:
    """
    Load and cache the vocab artefact. Returns a dict with at least keys:
      - "bloom": TinyBloom instance
      - "df": mapping term -> document frequency
    Raises a helpful error if the file is missing or malformed.
    """
    global _vocab
    if _vocab is not None:
        return _vocab

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Vocab artefact not found at {p}. "
            f"Run tools/build_vocab.py to generate artifacts/vocab.pkl"
        )
    with p.open("rb") as f:
        obj = _CompatUnpickler(f).load()

    if not isinstance(obj, dict) or "bloom" not in obj:
        raise ValueError(
            f"Invalid vocab artefact at {p}: expected a dict with key 'bloom'."
        )

    _vocab = obj
    return _vocab

def norm_tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]

def _stem(tok: str) -> str:
    try:
        return _stemmer.stem(tok)
    except Exception:
        return tok


def keyword_hits(query: str, vocab_path: str = _VOCAB_PATH_DEFAULT) -> int:
    vocab = _ensure_vocab(vocab_path)
    bloom = vocab.get("bloom")
    if not bloom:
        return 0

    toks = norm_tokens_nostop(query)[:20]  # <-- no stopwords
    hits = 0
    for t in toks:
        if (t in bloom) or ((STEM_NS + _stem(t)) in bloom):
            hits += 1
    return hits

def lexical_coverage(query: str, vocab_path: str = _VOCAB_PATH_DEFAULT) -> float:
    """
    Fraction of >=3-char tokens that appear in the vocab (raw or stem).
    Useful to block gibberish even if the classifier is overeager.
    """
    vocab = _ensure_vocab(vocab_path)
    bloom = vocab.get("bloom")
    if not bloom:
        return 0.0

    toks = norm_tokens_nostop(query)[:32]
    if not toks:
        return 0.0

    def has(tok: str) -> bool:
        return (tok in bloom) or ((STEM_NS + _stem(tok)) in bloom)

    hits = sum(1 for t in toks if has(t))
    return hits / max(len(toks), 1)

def idf_score(query: str, vocab_path: str = _VOCAB_PATH_DEFAULT) -> float:
    """
    Sum of IDF over non-stopword tokens that appear in the vocab.
    IDF = log((N+1)/(df(token)+1)), where N = docs_seen in the artefact.
    """
    vocab = _ensure_vocab(vocab_path)
    bloom = vocab.get("bloom")
    df_map = vocab.get("df") or {}
    N = int(vocab.get("docs_seen") or 1)

    toks = norm_tokens_nostop(query)[:32]
    if not toks or not bloom:
        return 0.0

    total = 0.0
    for t in toks:
        # count only tokens we know (raw or stem)
        known = (t in bloom) or ((STEM_NS + _stem(t)) in bloom)
        if not known:
            continue
        df = int(df_map.get(t, 0))
        total += float(__import__("math").log((N + 1) / (df + 1)))
    return total

def fuzzy_bonus(query: str, top_k: int = 50) -> int:
    """Award small extra hits for close matches to rare terms (typo tolerance)."""
    try:
        vocab = _ensure_vocab()  # uses default path
    except Exception:
        return 0

    df_obj: Any = vocab.get("df")
    if not df_obj:
        return 0

    # Normalise to a plain dict[str, int]
    try:
        df: Dict[str, int] = dict(df_obj)
    except Exception:
        return 0

    if not df:
        return 0

    # rare terms ≈ low DF (≈ high IDF)
    rare_terms = [k for k, _ in heapq.nsmallest(top_k, df.items(), key=lambda kv: kv[1])]

    q = " ".join(norm_tokens(query))
    close = 0
    for term in rare_terms[:10]:
        if token_set_ratio(q, str(term)) >= 85:
            close += 1
    return close
