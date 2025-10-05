from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceSpanPayload(BaseModel):
    """Sentence-level evidence span produced by Tier-3 models."""

    model_config = ConfigDict(extra="ignore")

    section_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    sentence_index: Optional[int] = Field(default=None, ge=0)
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ensure_span_consistency(self) -> "EvidenceSpanPayload":
        if self.start is None and self.end is None:
            return self
        if self.start is None or self.end is None or self.start > self.end:
            msg = "Evidence spans must include both start and end offsets"
            raise ValueError(msg)
        return self


class NumericRangePayload(BaseModel):
    """Normalized numeric range associated with a measurement."""

    model_config = ConfigDict(extra="ignore")

    minimum: float = Field(..., description="Lower bound of the reported range")
    maximum: float = Field(..., description="Upper bound of the reported range")

    @model_validator(mode="after")
    def _ensure_order(self) -> "NumericRangePayload":
        if self.minimum > self.maximum:
            msg = "Numeric range minimum must be <= maximum"
            raise ValueError(msg)
        return self


class NumericMeasurementPayload(BaseModel):
    """Structured representation of a numeric result extracted in Tier-3."""

    model_config = ConfigDict(extra="ignore")

    metric: str = Field(..., min_length=2, max_length=120)
    value: float = Field(..., description="Primary numeric value extracted from evidence")
    value_text: str = Field(..., min_length=1, max_length=48)
    unit: Optional[str] = Field(default=None, max_length=16)
    dataset: Optional[str] = Field(default=None, max_length=160)
    split: Optional[str] = Field(default=None, max_length=60)
    task: Optional[str] = Field(default=None, max_length=160)
    confidence_interval: Optional[str] = Field(default=None, max_length=64)
    normalization_hash: Optional[str] = Field(default=None, min_length=8, max_length=64)
    value_range: Optional[NumericRangePayload] = Field(default=None)

    @field_validator("metric")
    @classmethod
    def _normalize_metric(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "Metric name cannot be empty"
            raise ValueError(msg)
        return cleaned

    @field_validator("unit")
    @classmethod
    def _normalize_unit(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        replacements = {
            "percent": "%",
            "percentage": "%",
            "points": "points",
            "pt": "points",
        }
        lowered = cleaned.lower()
        if lowered in replacements:
            return replacements[lowered]
        return cleaned


class Tier3RelationCandidate(BaseModel):
    """Validated Tier-3 relation candidate returned by LLM fallbacks."""

    model_config = ConfigDict(extra="ignore")

    subject: str = Field(..., min_length=2, max_length=160)
    relation: str = Field(..., min_length=2, max_length=80)
    object: str = Field(..., min_length=1, max_length=200)
    evidence: str = Field(..., min_length=10, max_length=800)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list, max_length=16)
    triple_conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    schema_match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    provenance: Optional[str] = Field(default=None, max_length=120)


__all__ = [
    "EvidenceSpanPayload",
    "NumericMeasurementPayload",
    "NumericRangePayload",
    "Tier3RelationCandidate",
]

