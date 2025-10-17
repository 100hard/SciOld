from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, cast
from uuid import UUID

from app.db.pool import get_pool
from app.models.compare import (
    ComparisonCell,
    ComparisonEntitySummary,
    ComparisonPaper,
    ComparisonRow,
    ComparisonSummary,
    PaperComparisonResponse,
)


class PaperComparisonError(ValueError):
    """Raised when the comparison request cannot be fulfilled."""


async def compare_papers(paper_ids: Sequence[UUID]) -> PaperComparisonResponse:
    """Build a comparison across papers without assuming a particular domain.

    Every result row is interpreted using the canonical method, dataset, metric,
    and task tables so multi-disciplinary corpora (e.g., chemistry, biology,
    physics, or ML) are handled uniformly. The function simply aggregates the
    normalized entities already stored for each paper, which keeps the workflow
    domain agnostic.
    """

    if len(paper_ids) < 2:
        raise PaperComparisonError("At least two paper identifiers are required for comparison.")

    unique_ids = list(dict.fromkeys(paper_ids))
    pool = get_pool()

    async with pool.acquire() as conn:
        paper_rows = await conn.fetch(
            "SELECT id, title, year FROM papers WHERE id = ANY($1::uuid[])",
            unique_ids,
        )

        found_papers: Dict[UUID, Mapping[str, object]] = {row["id"]: dict(row) for row in paper_rows}
        missing = [pid for pid in unique_ids if pid not in found_papers]
        if missing:
            missing_str = ", ".join(str(pid) for pid in missing)
            raise PaperComparisonError(f"Unknown paper identifiers: {missing_str}")

        result_rows = await conn.fetch(
            """
            SELECT
                r.id AS result_id,
                r.paper_id,
                r.method_id,
                m.name AS method_name,
                r.dataset_id,
                d.name AS dataset_name,
                r.metric_id,
                mt.name AS metric_name,
                mt.unit AS metric_unit,
                r.task_id,
                t.name AS task_name,
                r.split,
                r.value_numeric,
                r.value_text,
                r.unit,
                r.is_sota,
                r.confidence,
                r.evidence
            FROM results r
            LEFT JOIN methods m ON r.method_id = m.id
            LEFT JOIN datasets d ON r.dataset_id = d.id
            LEFT JOIN metrics mt ON r.metric_id = mt.id
            LEFT JOIN tasks t ON r.task_id = t.id
            WHERE r.paper_id = ANY($1::uuid[])
            ORDER BY COALESCE(d.name, ''), COALESCE(mt.name, ''), COALESCE(m.name, ''), COALESCE(r.split, '')
            """,
            unique_ids,
        )

    papers: List[ComparisonPaper] = []
    for pid in unique_ids:
        paper_data = found_papers[pid]
        title = cast(Optional[str], paper_data.get("title"))
        year = cast(Optional[int], paper_data.get("year"))
        papers.append(ComparisonPaper(id=pid, title=title, year=year))

    entity_sets: Dict[UUID, Dict[str, Set[str]]] = {
        pid: {
            "methods": set(),
            "datasets": set(),
            "metrics": set(),
            "tasks": set(),
        }
        for pid in unique_ids
    }

    matrix_map: MutableMapping[
        Tuple[UUID | None, UUID | None, UUID | None, UUID | None, str | None],
        ComparisonRow,
    ] = {}

    for row in result_rows:
        paper_id: UUID = row["paper_id"]
        method_name = row["method_name"]
        dataset_name = row["dataset_name"]
        metric_name = row["metric_name"]
        task_name = row["task_name"]

        if method_name:
            entity_sets[paper_id]["methods"].add(method_name)
        if dataset_name:
            entity_sets[paper_id]["datasets"].add(dataset_name)
        if metric_name:
            entity_sets[paper_id]["metrics"].add(metric_name)
        if task_name:
            entity_sets[paper_id]["tasks"].add(task_name)

        key = (
            row["method_id"],
            row["dataset_id"],
            row["metric_id"],
            row["task_id"],
            row["split"],
        )

        if key not in matrix_map:
            matrix_map[key] = ComparisonRow(
                method_id=row["method_id"],
                method_name=method_name,
                dataset_id=row["dataset_id"],
                dataset_name=dataset_name,
                metric_id=row["metric_id"],
                metric_name=metric_name,
                metric_unit=row["metric_unit"],
                task_id=row["task_id"],
                task_name=task_name,
                split=row["split"],
                papers={},
            )

        numeric_value = row["value_numeric"]
        confidence_value = row["confidence"]
        matrix_map[key].papers[paper_id] = ComparisonCell(
            result_id=row["result_id"],
            value_numeric=float(numeric_value) if numeric_value is not None else None,
            value_text=row["value_text"],
            unit=row["unit"],
            is_sota=row["is_sota"],
            confidence=float(confidence_value) if confidence_value is not None else None,
            evidence=list(row["evidence"] or []),
        )

    def build_entity_summary(field: str) -> ComparisonEntitySummary:
        shared_values = _shared_entities(entity_sets.values(), field)
        unique_values = _unique_entities(entity_sets, field)
        return ComparisonEntitySummary(shared=shared_values, unique=unique_values)

    summary = ComparisonSummary(
        methods=build_entity_summary("methods"),
        datasets=build_entity_summary("datasets"),
        metrics=build_entity_summary("metrics"),
        tasks=build_entity_summary("tasks"),
    )

    matrix = sorted(
        matrix_map.values(),
        key=lambda row: (
            (row.dataset_name or "").lower(),
            (row.metric_name or "").lower(),
            (row.method_name or "").lower(),
            (row.task_name or "").lower(),
            row.split or "",
        ),
    )

    return PaperComparisonResponse(papers=papers, summary=summary, matrix=matrix)


def _shared_entities(entity_sets: Iterable[Mapping[str, Set[str]]], field: str) -> List[str]:
    sets = [values[field] for values in entity_sets]
    if not sets:
        return []
    shared = set.intersection(*sets)  # type: ignore[arg-type]
    return sorted(shared)


def _unique_entities(entity_sets: Mapping[UUID, Mapping[str, Set[str]]], field: str) -> Dict[UUID, List[str]]:
    unique: Dict[UUID, List[str]] = {}
    for paper_id, values in entity_sets.items():
        current = set(values[field])
        other_values: Set[str] = set()
        for other_id, other in entity_sets.items():
            if other_id == paper_id:
                continue
            other_values.update(other[field])
        unique_values = sorted(current - other_values)
        unique[paper_id] = unique_values
    return unique
