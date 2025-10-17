from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.compare import PaperComparisonResponse
from app.services.compare import PaperComparisonError, compare_papers

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("", response_model=PaperComparisonResponse)
async def compare_papers_endpoint(papers: str = Query(..., description="Comma-separated list of paper UUIDs")) -> PaperComparisonResponse:
    paper_ids = _parse_paper_ids(papers)
    try:
        return await compare_papers(paper_ids)
    except PaperComparisonError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_paper_ids(raw: str) -> List[UUID]:
    segments = [segment.strip() for segment in raw.split(",") if segment.strip()]
    if not segments:
        raise HTTPException(status_code=400, detail="At least two paper identifiers must be provided.")

    try:
        paper_ids = [UUID(segment) for segment in segments]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="One or more paper identifiers are invalid UUIDs.") from exc

    if len(paper_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two paper identifiers must be provided.")

    return paper_ids
