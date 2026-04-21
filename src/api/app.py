from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
STATIC_DIR = ROOT / "src" / "frontend" / "static"

app = FastAPI(title="Instagram Engagement Analysis")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS), name="artifacts")


class PredictionRequest(BaseModel):
    followers: int = Field(..., ge=100, le=2_000_000)
    post_type: str
    caption_length: int = Field(..., ge=0, le=500)
    hashtags_count: int = Field(..., ge=0, le=30)
    hour_posted: int = Field(..., ge=0, le=23)
    is_weekend: bool
    sponsorship: bool


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_models():
    likes_model = joblib.load(ARTIFACTS / "best_likes_model.joblib")
    comments_model = joblib.load(ARTIFACTS / "best_comments_model.joblib")
    return likes_model, comments_model


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/insights")
def insights() -> dict:
    return _load_json(ARTIFACTS / "summary.json")


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    likes_model, comments_model = _load_models()
    row = pd.DataFrame(
        [
            {
                "followers": payload.followers,
                "post_type": payload.post_type,
                "caption_length": payload.caption_length,
                "hashtags_count": payload.hashtags_count,
                "hour_posted": payload.hour_posted,
                "is_weekend": int(payload.is_weekend),
                "sponsorship": int(payload.sponsorship),
            }
        ]
    )
    predicted_likes = max(0, round(float(likes_model.predict(row)[0])))
    predicted_comments = max(0, round(float(comments_model.predict(row)[0])))

    return {
        "predicted_likes": predicted_likes,
        "predicted_comments": predicted_comments,
        "recommendation": (
            "Reels and carousels usually perform best in this dataset. "
            "Try posting in the evening with a descriptive caption and a moderate number of hashtags."
        ),
    }
