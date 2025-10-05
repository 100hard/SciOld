from __future__ import annotations

import json
from typing import Any, Sequence
from uuid import UUID

try:
    from asyncpg.pgproto import pgproto  # type: ignore
except Exception:  # pragma: no cover - asyncpg may be absent
    pgproto = None  # type: ignore[assignment]

from app.db.pool import get_pool
from app.models.triple_candidate import TripleCandidateRecord


INSERT_SQL = """
    INSERT INTO triple_candidates (
        paper_id,
        section_id,
        subject_text,
        relation_text,
        object_text,
        subject_span,
        object_span,
        subject_type_guess,
        relation_type_guess,
        object_type_guess,
        evidence_text,
        triple_conf,
        schema_match_score,
        tier,
        graph_metadata,
        provenance
    )
    VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
    )
"""


def _encode_json(value: object) -> object:
    if value is None:
        return None
    jsonb_factory = getattr(pgproto, "Jsonb", None) if pgproto is not None else None
    if jsonb_factory is not None:
        return jsonb_factory(value)
    return json.dumps(value)


async def replace_triple_candidates(
    paper_id: UUID,
    candidates: Sequence[TripleCandidateRecord],
) -> None:
    """Replace stored triple candidates for a paper with the latest payload."""

    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM triple_candidates WHERE paper_id = $1",
                paper_id,
            )
            if not candidates:
                return

            section_lookup_needed = any(candidate.section_id for candidate in candidates)
            valid_section_ids: set[UUID] = set()
            if section_lookup_needed:
                rows = await conn.fetch(
                    "SELECT id FROM sections WHERE paper_id = $1",
                    paper_id,
                )
                valid_section_ids = {row["id"] for row in rows}

            records = []
            for candidate in candidates:
                section_uuid = _safe_uuid(candidate.section_id)
                if section_uuid and section_uuid not in valid_section_ids:
                    section_uuid = None

                provenance_payload = dict(candidate.provenance or {})
                verifier_details: dict[str, Any] = {}
                if candidate.verification:
                    verifier_details["verification"] = candidate.verification
                if candidate.confidence_components:
                    verifier_details["confidence_components"] = candidate.confidence_components
                if candidate.verifier_notes:
                    verifier_details["notes"] = candidate.verifier_notes
                if verifier_details:
                    existing = provenance_payload.get("verifier")
                    if isinstance(existing, dict):
                        existing.update({k: v for k, v in verifier_details.items() if v})
                    else:
                        provenance_payload["verifier"] = verifier_details

                records.append(
                    (
                        candidate.paper_id,
                        section_uuid,
                        candidate.subject,
                        candidate.relation,
                        candidate.object,
                        candidate.subject_span,
                        candidate.object_span,
                        candidate.subject_type_guess,
                        candidate.relation_type_guess,
                        candidate.object_type_guess,
                        candidate.evidence,
                        candidate.triple_conf,
                        candidate.schema_match_score,
                        candidate.tier,
                        candidate.graph_metadata,
                        provenance_payload,

                    )
                )

            await conn.executemany(INSERT_SQL, records)


def _safe_uuid(value: str | None) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None
