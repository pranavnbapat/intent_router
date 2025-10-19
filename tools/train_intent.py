# tools/train_intent.py

import json
import pickle
import re

from pathlib import Path

from typing import Iterable, Dict, Any, List
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")

base = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")


def latest_input_file() -> Path:
    root = Path(__file__).resolve().parent.parent
    in_dir = root / "input"
    cands = sorted(in_dir.glob("final_output_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No final_output_*.json in {in_dir}")
    return cands[0]

def read_records(path: Path) -> Iterable[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8").lstrip()
    if txt.startswith("["):
        arr = json.loads(txt)
        if isinstance(arr, list):
            for o in arr:
                if isinstance(o, dict): yield o
        return
    for line in txt.splitlines():
        line = line.strip()
        if not line: continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if isinstance(o, dict): yield o

def ko_text(o: Dict[str, Any]) -> str:
    parts = [
        o.get("title", ""),
        o.get("description", ""),
        o.get("ko_content_flat", ""),
    ]
    return " \n ".join(map(str, parts))

def load_negatives() -> List[str]:
    p = Path(__file__).resolve().parent / "neg_samples.txt"
    if not p.exists():
        # minimal fallback
        return [
            "what is the weather today in london",
            "how to fix a laptop battery not charging",
            "what is the capital of france",
            "latest iphone price and features",
            "best programming language for beginners",
            "how to reset my wifi router at home",
        ]
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    src = latest_input_file()
    print(f"[train_intent] Using positives from: {src}")

    pos = [ko_text(o) for o in read_records(src)]
    neg = load_negatives()

    # Defensive: if negatives are too few, add built-in fallbacks
    if len(neg) < 50:
        neg += [
                   "what is the weather today in amsterdam",
                   "how to reset a wifi router",
                   "latest iphone price and features",
                   "football match schedule tonight",
                   "best programming language for beginners",
                   "train times from utrecht to rotterdam",
                   "cheap flights to rome",
                   "what time is sunset",
                   "netflix subscription price",
                   "how to boil pasta"
               ] * 10  # replicate to get count up

    X = pos + neg
    y = [1] * len(pos) + [0] * len(neg)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=1,  # be permissive; tiny datasets need this
            max_df=0.98,
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
    ])

    pipe.fit(Xtr, ytr)

    print(classification_report(yte, pipe.predict(Xte), zero_division=0))

    out = Path(__file__).resolve().parent.parent / "artifacts" / "intent_clf.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(pipe, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[train_intent] Saved {out}")

if __name__ == "__main__":
    main()
