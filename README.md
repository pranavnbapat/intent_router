# Intent Router (Lexical + Intent)

A fast router that decides whether to:
- run RAG (with custom prompt) or
- go LLM-only (send the user input straight with a generic prompt)

It uses:
- a deterministic Bloom vocabulary built from your KO corpus (raw + stemmed)
- a lightweight intent classifier (TF-IDF + Logistic Regression with probability calibration)

## Install
```
git clone https://github.com/pranavnbapat/intent_router.git
cd intent_router
pip install -r requirements.txt
```

## Use
### 1) Build EU Stopwords Bundle
Creates artifacts/stopwords_eu.pkl with per-language sets and a multilingual union.
```
python tools/build_stopwords_eu.py
```
Expected: [stopwords] Saved artifacts/stopwords_eu.pkl with <N> languages, union size=<~10k>

### 2) Build Vocabulary (Bloom + DF)
Parses the latest input/final_output_*.json, extracts title, description, ko_content_flat, 
etc., tokenises, stems (Snowball/English), and builds a deterministic Bloom filter plus 
per-token document frequency.
```
python tools/build_vocab.py
```
**Outputs:**
artifacts/vocab.pkl containing:
- bloom (raw tokens + §stem namespace),
- df (document frequency map),
- docs_seen,
- metadata.

**Why deterministic?**
The Bloom uses hashlib.blake2b (not Python’s hash()), so membership is stable across processes.

### 3) Train Intent Classifier
Trains a calibrated LR classifier (TF-IDF n-grams) to predict in-domain (agri) vs out-of-domain.
```
python tools/train_intent.py
```

**Inputs:**
- Positives: KO texts from the same snapshot as step 2.
- Negatives: tools/neg_samples.txt (provide hundreds–thousands of generic, non-agri lines).

**Outputs:**
```artifacts/intent_clf.pkl```

Tip: Grow neg_samples.txt over time (news, shopping, travel, chit-chat).
This reduces over-confidence on junk inputs.

### 4) Run the Router (CLI)
**Example 1:**
```
python run_router.py "farm crop"
```
```
{
  "path": "RAG",
  "hits": 2,
  "p_intent": 0.9989308007418334,
  "tokens": 2,
  "notes": {
    "short": true,
    "coverage": 1.0,
    "idf": 1.988
  }
}
```

**Example 2:**
```
python run_router.py "good morning"
```
```
{
  "path": "LLM_ONLY",
  "hits": 0,
  "p_intent": 0.0,
  "tokens": 1,
  "notes": {
    "short": true,
    "nonsense": true
  }
}
```

**Example 3:**
```
python run_router.py "shdbf7q4 tfaoi..."
```
```
{
  "path": "LLM_ONLY",
  "hits": 0,
  "p_intent": 0.9986196830650673,
  "tokens": 2,
  "notes": {
    "short": true,
    "coverage": 0.0,
    "idf": 0.0
  }
}
```

## Configuration (thresholds)
In ```router.py```:
```
MIN_HITS = 1            # min lexical hits
MIN_INTENT_PROB = 0.60  # raise to 0.65–0.70 when negatives improve
SHORT_MIN_PROB = 0.70

COVERAGE_MIN = 0.30     # 0.20 for very short queries (see short_gate)
IDF_MIN = 2.0
IDF_STRONG = 4.0
```
“Short” is currently ```len(norm_tokens(query)) <= 3```.

We also gate the fuzzy bonus so it can’t push junk into RAG:
