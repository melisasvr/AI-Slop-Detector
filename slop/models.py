from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["Likely Real", "Suspicious", "Likely AI Slop"]


def verdict_for(score: int) -> Verdict:
    """Single source of truth: the verdict is derived from the score, never trusted from the model."""
    return "Likely Real" if score < 35 else "Suspicious" if score < 65 else "Likely AI Slop"


class Evidence(BaseModel):
    signal: str
    source: Literal["transcript", "frames", "metadata", "signals"]
    detail: str
    points_to: Literal["slop", "authentic", "neutral"]
    strength: Literal["weak", "moderate", "strong"]


class Analysis(BaseModel):
    slop_score: int = Field(ge=0, le=100)
    verdict: Verdict
    confidence: int = Field(ge=0, le=100)
    summary: str
    evidence: list[Evidence] = Field(min_length=1)
    limitations: list[str] = []
