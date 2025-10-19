# router.py

import os

from dataclasses import dataclass
from dotenv import load_dotenv

from classifier import intent_score
from lexical import keyword_hits, fuzzy_bonus, norm_tokens, lexical_coverage, idf_score, norm_tokens_nostop

load_dotenv()

def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _get_env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default

def _get_env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return default if val is None or val.strip() == "" else val.strip()

VOCAB_PATH         = _get_env_str("VOCAB_PATH", "artifacts/vocab.pkl")
INTENT_MODEL_PATH  = _get_env_str("INTENT_MODEL_PATH", "artifacts/intent_clf.pkl")

MIN_HITS           = _get_env_int("MIN_HITS", 2)
MIN_INTENT_PROB    = _get_env_float("MIN_INTENT_PROB", 0.75)
SHORT_MIN_PROB     = _get_env_float("SHORT_MIN_PROB", 0.80)

SHORT_TOK_MIN      = _get_env_int("SHORT_TOK_MIN", 2)
MIN_MATCHES        = _get_env_int("MIN_MATCHES", 2)

COVERAGE_MIN       = _get_env_float("COVERAGE_MIN", 0.35)
IDF_MIN            = _get_env_float("IDF_MIN", 3.0)
IDF_STRONG         = _get_env_float("IDF_STRONG", 5.0)


@dataclass
class Decision:
    path: str        # "RAG" | "LLM_ONLY"
    hits: int
    p_intent: float
    tokens: int      # number of (cleaned) query tokens
    notes: dict

def _looks_nonsense(q: str) -> bool:
    letters = sum(ch.isalpha() for ch in q)
    digits = sum(ch.isdigit() for ch in q)
    total = max(len(q), 1)
    alpha_ratio = letters / total

    toks_all = norm_tokens(q)  # includes stopwords
    toks_ns = norm_tokens_nostop(q)  # >=3 chars, no stopwords

    # Heuristics:
    # - too few tokens
    # - too many digits relative to letters
    # - mostly non-alphabetic chars
    # - any token length==1-2 after nostop cleanup (already filtered), so len check is enough
    return (
            len(toks_all) < 2
            or len(toks_ns) < 3
            or alpha_ratio < 0.75
            or (digits > 0 and (digits / max(letters, 1)) > 0.25)
    )


def route(query: str) -> Decision:
    if _looks_nonsense(query):
        return Decision(
            path="LLM_ONLY",
            hits=0,
            p_intent=0.0,
            tokens=len(norm_tokens_nostop(query)),
            notes={"short": True, "nonsense": True}
        )

    toks = norm_tokens_nostop(query)

    hits = keyword_hits(query, VOCAB_PATH)
    # Only allow a small fuzzy boost when we already have lexical evidence
    if hits >= 1 and len(norm_tokens(query)) >= 3 and query.isalpha():
        hits += min(1, fuzzy_bonus(query))

    p = intent_score(query, INTENT_MODEL_PATH)
    short = len(norm_tokens(query)) <= 3

    cov = lexical_coverage(query, VOCAB_PATH)
    idf = idf_score(query, VOCAB_PATH)

    # Hard junk guard: if we have *no* lexical evidence, don't trust the classifier.
    if hits == 0 and (cov < 0.20 or idf < 2.0):
        return Decision(
            path="LLM_ONLY",
            hits=hits,
            p_intent=p,
            tokens=len(toks),
            notes={"short": short, "coverage": round(cov, 3), "idf": round(idf, 3), "junk_guard": True},
        )

    # Short queries must be truly lexical: at least 2 hits AND minimal coverage/idf
    short_gate = (
            short
            and hits >= 2
            and cov >= 0.30
            and idf >= IDF_MIN
            and p >= SHORT_MIN_PROB
    )

    # Main gate: lexical + model, with either coverage or IDF backing it
    main_gate = (
            hits >= MIN_HITS
            and p >= MIN_INTENT_PROB
            and cov >= COVERAGE_MIN
            and idf >= IDF_MIN
    )

    # Classifier override only if (a) not short, (b) very confident, (c) decent coverage, (d) at least one lexical hit
    clf_override = (
            not short
            and p >= 0.93
            and hits >= 2
            and cov >= COVERAGE_MIN
            and idf >= IDF_STRONG
    )

    use_rag = main_gate or short_gate or clf_override

    return Decision(
        path="RAG" if use_rag else "LLM_ONLY",
        hits=hits,
        p_intent=p,
        tokens=len(toks),
        notes={"short": short, "coverage": round(cov, 3), "idf": round(idf, 3)},
    )
