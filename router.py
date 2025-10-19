# router.py

from dataclasses import dataclass

from classifier import intent_score
from lexical import keyword_hits, fuzzy_bonus, norm_tokens, lexical_coverage, idf_score, norm_tokens_nostop


VOCAB_PATH = "artifacts/vocab.pkl"
INTENT_MODEL_PATH = "artifacts/intent_clf.pkl"

MIN_HITS = 1            # lexical gate
MIN_INTENT_PROB = 0.65  # classifier threshold
SHORT_MIN_PROB = 0.70   # higher bar for very short queries

# require at least this many non-stop tokens for short gate
SHORT_TOK_MIN = 2

MIN_MATCHES = 2

COVERAGE_MIN = 0.30   # require at least 20% lexical coverage for RAG
IDF_MIN = 2.0           # sum of IDF needed for weak queries
IDF_STRONG = 4.0        # stronger bar for classifier override

@dataclass
class Decision:
    path: str        # "RAG" | "LLM_ONLY"
    hits: int
    p_intent: float
    tokens: int      # number of (cleaned) query tokens
    notes: dict

def _looks_nonsense(q: str) -> bool:
    letters = sum(ch.isalpha() for ch in q)
    total   = max(len(q), 1)
    alpha_ratio = letters / total

    toks_all = norm_tokens(q)            # includes stopwords
    toks_ns  = norm_tokens_nostop(q)     # >=3 chars, no stopwords

    return (
        len(toks_all) < 2
        or len(toks_ns) < 2
        or alpha_ratio < 0.6
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
    if hits >= 1 and len(norm_tokens(query)) >= 3:
        hits += min(1, fuzzy_bonus(query))

    p = intent_score(query, INTENT_MODEL_PATH)
    short = len(norm_tokens(query)) <= 3

    cov = lexical_coverage(query, VOCAB_PATH)
    idf = idf_score(query, VOCAB_PATH)

    # Short queries must be truly lexical: at least 2 hits AND minimal coverage/idf
    short_gate = (
            short
            and hits >= 2
            and cov >= 0.20
            and idf >= IDF_MIN
            and p >= SHORT_MIN_PROB
    )

    # Main gate: lexical + model, with either coverage or IDF backing it
    main_gate = (
            hits >= MIN_HITS
            and p >= MIN_INTENT_PROB
            and (cov >= COVERAGE_MIN or idf >= IDF_MIN)
    )

    # Classifier override only if (a) not short, (b) very confident, (c) decent coverage, (d) at least one lexical hit
    clf_override = (
            not short
            and p >= 0.93
            and hits >= 1
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
