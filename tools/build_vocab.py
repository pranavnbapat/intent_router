# tools/build_vocab.py

import json
import os
import pickle
import re
import sys

from collections import Counter
from pathlib import Path
from typing import Iterable, Dict, Any, List

from nltk.stem.snowball import SnowballStemmer


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinybloom import TinyBloom

# ---------------------------- Config & constants ----------------------------
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")  # basic unicode letters

def _load_stopwords_bundle() -> dict:
    """
    Load the EU stopwords bundle once (lang_code -> set[str], plus 'all_union').
    """
    candidates = [
        Path("artifacts/stopwords_eu.pkl"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("stopwords_eu.pkl not found in artefacts/ or artifacts/")


_STOP_BUNDLE = None
def stop_set(lang_code: str | None = "en") -> set[str]:
    """
    Return the stopword set. If the language is unknown, we default to 'all_union'
    so the filter still works reasonably across languages.
    """
    global _STOP_BUNDLE
    if _STOP_BUNDLE is None:
        _STOP_BUNDLE = _load_stopwords_bundle()
    return _STOP_BUNDLE.get(lang_code or "en", _STOP_BUNDLE.get("all_union", set()))

# Use the union by default for vocabulary building (multilingual corpus)
STOP = stop_set("all_union")

DEFAULT_OUT = "artifacts/vocab.pkl"  # prefer 'artefacts' folder


# ---------------------------- Input discovery & reading ----------------------------
def get_latest_json_file() -> str:
    """
    Find the latest snapshot inside
    """
    root = Path(__file__).resolve().parent.parent
    in_dir = root / "input"
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    candidates = sorted(
        [p for p in in_dir.rglob("final_output_*.json") if p.is_file() and not p.name.endswith(".tmp")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No JSON files found under {in_dir} matching final_output_*.json")
    return str(candidates[0])


def read_records(path: str) -> Iterable[Dict[str, Any]]:
    """
    Supports:
      - JSON array: [ {...}, {...}, ... ]
      - NDJSON: one JSON object per line
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8").lstrip()

    if txt.startswith("["):
        # JSON array
        try:
            arr = json.loads(txt)
        except Exception:
            arr = []
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
        return

    # NDJSON fallback
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


# TinyBloom.__module__ = "tinybloom"

# ---------------------------- Normalisation ----------------------------
def normalise(text: str):
    return [w.lower() for w in WORD.findall(text)]


# ---------------------------- Main ----------------------------
def main():
    latest = get_latest_json_file()
    print(f"[build_vocab] Using snapshot: {latest}")

    # English stemmer for now; swap to multilingual
    stemmer = SnowballStemmer("english")

    raw_vocab, stem_vocab = set(), set()
    df_counter = Counter()   # document frequency per raw token
    docs = 0

    for obj in read_records(latest):
        docs += 1

        # Pick fields we care about
        fields = [
            obj.get("title", ""),
            obj.get("description", ""),
            obj.get("ko_content_flat", ""),
            obj.get("project_display_name", ""),
            obj.get("project_name", ""),
            obj.get("project_acronym", ""),
            obj.get("keywords", []),
            obj.get("topics", []),
        ]

        # keywords may be list/dict → flatten
        flat: List[str] = []
        for f in fields:
            if isinstance(f, list):
                flat.extend(map(str, f))
            elif isinstance(f, dict):
                flat.extend(map(str, f.values()))
            else:
                flat.append(str(f))
        text = " \n ".join(flat)

        # Token filter:
        #  - length > 2
        #  - not in multilingual stopwords union
        tokens = [t for t in normalise(text) if len(t) > 2]
        # tokens = [t for t in normalise(text) if len(t) > 2 and t not in STOP]

        # Use doc_uniques to count DF only once per document
        doc_uniques = set()
        for t in tokens:
            raw_vocab.add(t)
            doc_uniques.add(t)
            # Keep stem namespace separate; we stem only to widen matching later
            stem_vocab.add(stemmer.stem(t))
        for t in doc_uniques:
            df_counter[t] += 1

    # Build Bloom (namespaced for stems)
    bloom = TinyBloom(m_bits=4_000_000, k=7)
    for t in raw_vocab:
        bloom.add(t)
    for t in stem_vocab:
        bloom.add("§" + t)  # namespace stemmed entries

    # Persist artefact
    out_path = Path(os.getenv("VOCAB_OUT", DEFAULT_OUT))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bloom": bloom,
        "raw_vocab_size": len(raw_vocab),
        "stem_vocab_size": len(stem_vocab),
        "df": df_counter,
        "docs_seen": docs,
        "source_file": latest,
        "stopwords_info": {
            "source": "stopwords_eu.pkl",
            "scope": "all_union"
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(
        f"[build_vocab] Saved {out_path} | "
        f"docs={docs} raw={len(raw_vocab)} stems={len(stem_vocab)}"
    )


if __name__ == "__main__":
    main()
