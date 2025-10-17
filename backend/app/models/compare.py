from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ComparisonPaper(BaseModel):
    id: UUID
    title: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1800, le=2100)


class ComparisonEntitySummary(BaseModel):
    shared: List[str] = Field(default_factory=list)
    unique: Dict[UUID, List[str]] = Field(default_factory=dict)


class ComparisonSummary(BaseModel):
    methods: ComparisonEntitySummary
    datasets: ComparisonEntitySummary
    metrics: ComparisonEntitySummary
    tasks: ComparisonEntitySummary


class ComparisonCell(BaseModel):
    result_id: UUID
    value_numeric: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    is_sota: Optional[bool] = None
    confidence: Optional[float] = None
    evidence: List[Mapping[str, Any]] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    method_id: Optional[UUID] = None
    method_name: Optional[str] = None
    dataset_id: Optional[UUID] = None
    dataset_name: Optional[str] = None
    metric_id: Optional[UUID] = None
    metric_name: Optional[str] = None
    metric_unit: Optional[str] = None
    task_id: Optional[UUID] = None
    task_name: Optional[str] = None
    split: Optional[str] = None
    papers: Dict[UUID, ComparisonCell] = Field(default_factory=dict)


class PaperComparisonResponse(BaseModel):
    papers: List[ComparisonPaper]
    summary: ComparisonSummary
    matrix: List[ComparisonRow]

    class Config:
        json_encoders = {UUID: str}
