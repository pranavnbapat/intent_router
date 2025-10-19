# classifier.py

import pickle

from typing import Any, Optional
from pathlib import Path

_INTENT_MODEL: Optional[Any] = None
_DEFAULT_PATH = "artifacts/intent_clf.pkl"


def _ensure(path: str = _DEFAULT_PATH):
    """
    Load and cache the intent classifier once, returning the model.
    Raises FileNotFoundError if the pickle does not exist.
    """
    global _INTENT_MODEL

    if _INTENT_MODEL is not None:
        return _INTENT_MODEL

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Intent model not found at {p}. "
            f"Run tools/train_intent.py to generate {p}"
        )

    with p.open("rb") as f:
        model = pickle.load(f)

    # Basic sanity check
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Loaded object at {p} has no 'predict_proba' method.")

    _INTENT_MODEL = model
    return _INTENT_MODEL


def intent_score(text: str, model_path: str = _DEFAULT_PATH) -> float:
    """
    Return P(in-domain/agri) ∈ [0,1].
    If the model cannot be loaded, returns 0.0 defensively.
    """
    try:
        model = _ensure(model_path)
        return float(model.predict_proba([text])[0, 1])
    except Exception as e:
        # Log or print if you want to see why it failed
        # print(f"[intent_score] warning: {e}")
        return 0.0
