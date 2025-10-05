from __future__ import annotations

import copy
import logging
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from app.core.config import settings
from app.schemas.tier3 import Tier3RelationCandidate
from app.services import extraction_tier2 as tier2
from app.services.extraction_tier2 import (
    Tier2ValidationError,
    TripleSchemaError,
    validate_triples,
)

logger = logging.getLogger(__name__)

FALLBACK_SOURCE = "tier3_llm"
_DEFAULT_PROMPT = (
    "Rule-based extraction produced too few relation triples. Identify additional "
    "method, dataset, task, or metric relations that are explicitly supported by "
    "the evidence. Return up to 6 high-confidence triples that obey the provided "
    "JSON schema and avoid duplicates of the existing candidates."
)


def _non_fallback_rule_count(candidates: Sequence[dict[str, Any]]) -> int:
    count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source = str(candidate.get("source") or "").strip().lower()
        tier = str(candidate.get("tier") or "").strip().lower()
        if source == FALLBACK_SOURCE or tier == FALLBACK_SOURCE:
            continue
        count += 1
    return count


async def maybe_apply_relation_llm_fallback(
    paper_id: UUID,
    summary: dict[str, Any],
    *,
    enabled: Optional[bool] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Optionally invoke an LLM to backfill relation triples when rules lack coverage."""

    effective_enabled = settings.tier3_relation_fallback_enabled if enabled is None else bool(enabled)
    candidates = list(summary.get("triple_candidates") or [])
    sections = summary.get("sections") or []
    threshold = max(0, int(settings.tier3_relation_fallback_min_rule_candidates or 0))
    rule_count = _non_fallback_rule_count(candidates)

    meta: dict[str, Any] = {
        "enabled": effective_enabled,
        "triggered": False,
        "rule_candidates": rule_count,
        "threshold": threshold,
        "attempts": 0,
        "accepted": 0,
        "status": "disabled" if not effective_enabled else "skipped",
        "errors": [],
    }

    if not effective_enabled:
        return [], meta

    if rule_count >= threshold and threshold > 0:
        meta["status"] = "rule_coverage_sufficient"
        return [], meta

    if not sections:
        meta.update({"triggered": True, "status": "no_sections"})
        meta["errors"].append("No sections available for fallback context")
        return [], meta

    contexts = tier2._prepare_section_contexts(sections)
    if not contexts:
        meta.update({"triggered": True, "status": "no_context"})
        meta["errors"].append("Unable to derive section contexts for fallback")
        return [], meta

    messages = tier2._build_messages(contexts)
    prompt = settings.tier3_relation_fallback_prompt or _DEFAULT_PROMPT
    if prompt:
        for message in reversed(messages):
            if message.get("role") == "user":
                existing = message.get("content") or ""
                message["content"] = f"{existing.rstrip()}\n\n{prompt.strip()}"
                break

    max_attempts = max(1, int(settings.tier3_relation_fallback_max_attempts or 1))
    base_temperature = settings.tier3_relation_fallback_temperature

    pending_messages: Sequence[dict[str, str]] = list(messages)
    repair_context = None
    temperature_override = base_temperature
    errors: list[str] = []

    for attempt in range(max_attempts):
        meta.update({"triggered": True, "attempts": attempt + 1, "status": "in_progress"})
        try:
            raw_content = await tier2._invoke_llm(
                pending_messages, temperature=temperature_override
            )
        except Tier2ValidationError as exc:
            message = f"LLM request failed on attempt {attempt + 1}: {exc}"
            logger.warning("[tier3] %s", message)
            errors.append(message)
            pending_messages = list(messages)
            repair_context = None
            temperature_override = base_temperature
            continue

        try:
            candidate_payload = tier2._parse_json(raw_content)
            if repair_context is not None:
                candidate_payload = tier2._apply_repair_patch(repair_context, candidate_payload)
                repair_context = None
                pending_messages = list(messages)
                temperature_override = base_temperature
        except Tier2ValidationError as exc:
            message = f"Fallback payload parse failed on attempt {attempt + 1}: {exc}"
            logger.warning("[tier3] %s", message)
            errors.append(message)
            continue

        try:
            validated = validate_triples(candidate_payload)
        except TripleSchemaError as exc:
            message = (
                f"Fallback payload schema violations on attempt {attempt + 1}: {exc}"
            )
            logger.warning("[tier3] %s", message)
            errors.append(message)
            if attempt + 1 >= max_attempts:
                break
            try:
                pending_messages, repair_context, temperature_override = tier2._build_repair_messages(
                    candidate_payload, exc.issues
                )
            except Tier2ValidationError as repair_exc:
                repair_message = f"Failed to build repair prompt: {repair_exc}"
                logger.warning("[tier3] %s", repair_message)
                errors.append(repair_message)
                repair_context = None
                break
            continue
        except Tier2ValidationError as exc:
            message = f"Fallback payload validation failed on attempt {attempt + 1}: {exc}"
            logger.warning("[tier3] %s", message)
            errors.append(message)
            pending_messages = list(messages)
            repair_context = None
            temperature_override = base_temperature
            continue

        triples: list[dict[str, Any]] = []
        for index, triple in enumerate(validated.triples):
            triple_dict = triple.model_dump()
            triple_dict["source"] = FALLBACK_SOURCE
            triple_dict["tier"] = FALLBACK_SOURCE
            triple_dict["provenance"] = FALLBACK_SOURCE
            triple_dict["retry_count"] = attempt
            triple_dict["fallback_attempt"] = attempt + 1
            triple_dict.setdefault("evidence_spans", [])
            triple_dict.setdefault(
                "candidate_id",
                f"{FALLBACK_SOURCE}_{paper_id.hex}_{attempt}_{index:03d}_{uuid4().hex[:8]}",
            )
            # Ensure Tier-3 specific payloads adhere to our schema contract.
            Tier3RelationCandidate.model_validate(triple_dict)
            triples.append(triple_dict)

        meta.update(
            {
                "status": "succeeded",
                "accepted": len(triples),
                "errors": list(dict.fromkeys(errors)),
                "retry_used": attempt,
            }
        )
        logger.info(
            "[tier3] LLM fallback produced %s triples for paper %s on attempt %s",
            len(triples),
            paper_id,
            attempt + 1,
        )
        return triples, meta

    meta.update(
        {
            "status": "failed",
            "errors": list(dict.fromkeys(errors)),
        }
    )
    return [], meta


async def run_tier3_relations(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Orchestrate Tier-3 relation processing for a paper."""

    if base_summary is None:
        raise ValueError("Tier-3 relations requires Tier-1 and Tier-2 summaries")

    summary = copy.deepcopy(base_summary)

    metadata_raw = summary.get("metadata")
    metadata: dict[str, Any]
    if isinstance(metadata_raw, dict):
        metadata = dict(metadata_raw)
    else:
        metadata = {}
    summary["metadata"] = metadata

    # Ensure the triple candidate list is materialized so we can mutate safely.
    existing_candidates = list(summary.get("triple_candidates") or [])
    summary["triple_candidates"] = existing_candidates

    # Track that Tier-3 processing has been attempted for this summary.
    tiers = set(summary.get("tiers") or [])
    tiers.update({1, 2, 3})
    summary["tiers"] = sorted(tiers)

    fallback_candidates: list[dict[str, Any]] = []
    fallback_meta: dict[str, Any]
    try:
        fallback_candidates, fallback_meta = await maybe_apply_relation_llm_fallback(
            paper_id,
            summary,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        fallback_meta = {
            "triggered": True,
            "status": "error",
            "errors": [str(exc)],
        }

    if fallback_candidates:
        existing_candidates.extend(fallback_candidates)

    tier3_meta = {
        "tier": "tier3_relations",
        "status": fallback_meta.get("status", "unknown"),
        "added_candidates": len(fallback_candidates),
        "total_candidates": len(existing_candidates),
        "fallback": fallback_meta,
    }

    metadata["tier3_relations"] = tier3_meta

    return summary


__all__ = [
    "FALLBACK_SOURCE",
    "maybe_apply_relation_llm_fallback",
    "run_tier3_relations",
]
