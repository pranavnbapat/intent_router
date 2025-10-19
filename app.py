# app.py

from __future__ import annotations

import time

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field

from tools import build_stopwords_eu as sw_mod
from tools import build_vocab as vocab_mod
from tools import train_intent as intent_mod

from router import route as route_query  # returns Decision dataclass


app = FastAPI(title="Intent Router API", version="1.0.0")


ARTIFACTS_DIR = Path("artifacts")
STOPWORDS_PKL = ARTIFACTS_DIR / "stopwords_eu.pkl"
VOCAB_PKL = ARTIFACTS_DIR / "vocab.pkl"
INTENT_PKL = ARTIFACTS_DIR / "intent_clf.pkl"
INPUT_DIR = Path("input")
INPUT_GLOB = "final_output_*.json"

# In-memory “prepared” flag + info
_PREPARED: bool = False
_PREPARED_AT: Optional[float] = None


# --------- Schemas ---------
class PrepareResponse(BaseModel):
    prepared: bool
    ran_now: bool
    prepared_at: Optional[float] = None
    stopwords_path: Optional[str] = None
    vocab_path: Optional[str] = None
    intent_path: Optional[str] = None
    message: Optional[str] = None


class IntentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User input text to route")
    # Optional toggles:
    force_prepare_if_needed: bool = Field(
        True,
        description="If artefacts are missing and /prepare hasn’t run, auto-run preparation once.",
    )


class IntentDecision(BaseModel):
    path: str
    hits: int
    p_intent: float
    tokens: int
    notes: dict


# --------- Helpers ---------
def _input_has_data() -> bool:
    """Return True if there is at least one input snapshot present."""
    return any(INPUT_DIR.glob(INPUT_GLOB))


def _artifacts_exist() -> bool:
    return STOPWORDS_PKL.exists() and VOCAB_PKL.exists() and INTENT_PKL.exists()


def _mark_prepared():
    global _PREPARED, _PREPARED_AT
    _PREPARED = True
    _PREPARED_AT = time.time()


def _maybe_mark_prepared_from_disk():
    """If artefacts are on disk, trust them and mark prepared."""
    if _artifacts_exist():
        _mark_prepared()


def _run_prepare_pipeline() -> None:
    """
    Runs stopwords -> vocab -> intent in-process using your modules.
    Raises HTTPException on user-correctable errors (e.g., no input files).
    """
    # 1) input presence check (graceful error if empty)
    if not INPUT_DIR.exists() or not _input_has_data():
        raise HTTPException(
            status_code=400,
            detail=(
                "No input snapshots found. Place at least one file matching "
                f"'{INPUT_DIR}/{INPUT_GLOB}' (JSON array or NDJSON) and retry /prepare."
            ),
        )

    # 2) sequential build
    sw_mod.main()       # writes artifacts/stopwords_eu.pkl
    vocab_mod.main()    # reads latest input, writes artifacts/vocab.pkl
    intent_mod.main()   # reads latest input, writes artifacts/intent_clf.pkl

    # 3) sanity check + mark prepared
    if not _artifacts_exist():
        raise HTTPException(
            status_code=500,
            detail="Preparation finished but required artefacts are missing. Check logs.",
        )
    _mark_prepared()


# --------- Endpoints ---------
@app.get("/health")
def health():
    _maybe_mark_prepared_from_disk()
    return {
        "status": "ok",
        "prepared": _PREPARED,
        "prepared_at": _PREPARED_AT,
        "artifacts_exist": _artifacts_exist(),
    }


@app.post("/prepare", response_model=PrepareResponse)
def prepare(force: bool = Query(False, description="Force rebuild even if artefacts exist")):
    """
    Build stopwords, vocab, and train intent in one go.
    - If `force=false` and artefacts exist, we do nothing and return prepared=True.
    - If input is missing, return 400 with a helpful message.
    """
    global _PREPARED

    ran_now = False

    if force:
        _run_prepare_pipeline()
        ran_now = True
    else:
        if _artifacts_exist():
            _maybe_mark_prepared_from_disk()
        else:
            _run_prepare_pipeline()
            ran_now = True

    return PrepareResponse(
        prepared=_PREPARED,
        ran_now=ran_now,
        prepared_at=_PREPARED_AT,
        stopwords_path=str(STOPWORDS_PKL) if STOPWORDS_PKL.exists() else None,
        vocab_path=str(VOCAB_PKL) if VOCAB_PKL.exists() else None,
        intent_path=str(INTENT_PKL) if INTENT_PKL.exists() else None,
        message="OK" if _PREPARED else "Not prepared",
    )


@app.post("/intent-router", response_model=IntentDecision)
def intent_router(payload: IntentRequest = Body(...)):
    """
    Routes the query:
      - If not prepared (in-memory) or artefacts missing, optionally run /prepare first.
      - Then call your existing router.route() and return its Decision as JSON.
    """
    global _PREPARED

    # Ensure prepared (either by memory flag or by checking on-disk artefacts)
    if not _PREPARED:
        _maybe_mark_prepared_from_disk()

    if not _PREPARED:
        if payload.force_prepare_if_needed:
            _run_prepare_pipeline()
        else:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The router is not prepared yet. Run POST /prepare first "
                    "or call /intent-router with force_prepare_if_needed=true."
                ),
            )

    # Now route
    decision = route_query(payload.query)
    # Convert your dataclass to a dict for Pydantic
    return IntentDecision(**decision.__dict__)
