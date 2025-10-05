from __future__ import annotations

import copy
import hashlib
from decimal import Decimal
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from uuid import UUID

from app.models.ontology import MethodRelationCreate, MethodRelationType, ResultCreate
from app.schemas.tier3 import NumericMeasurementPayload
from app.services.ontology_store import (
    append_method_relations,
    append_results,
    ensure_dataset,
    ensure_method,
    ensure_metric,
    ensure_task,
)
from app.services.extraction_tier3_relations import (
    maybe_apply_relation_llm_fallback,
)
TIER_NAME = "tier3_verifier"
_BASE_CONFIDENCE_FALLBACK = 0.55
_JSON_VALID_BONUS = 0.05
_COREF_BONUS = 0.04
_TABLE_MATCH_BONUS = 0.1
_DUPLICATE_BONUS_STEP = 0.03
_DUPLICATE_BONUS_CAP = 0.09
_QUALITATIVE_RELATION_CONFIDENCE = 0.65
_DATASET_RELATION_GUESSES = {"EVALUATED_ON", "USES", "TRAINS_ON", "TESTED_ON"}
_TASK_RELATION_GUESSES = {"PROPOSES", "ADDRESSES", "SOLVES", "TARGETS"}

_GRAPH_DATASET_RELATIONS = {"EVALUATED_ON", "USES"}
_GRAPH_METRIC_RELATIONS = {"MEASURES", "REPORTS"}
_GRAPH_TASK_RELATIONS = {"PROPOSES"}

_PROVENANCE_PRIORS = {
    "table": 0.1,
    "rule": 0.05,
    "spacy": 0.02,
    "llm": -0.05,
}

_COREF_HEAD_PATTERN = re.compile(
    r"^(?:this|that|these|those|our|the|its|their|it|they)\b",
    re.IGNORECASE,
)
_COREF_NOUN_PATTERN = re.compile(
    r"(model|method|approach|system|framework|technique|architecture|algorithm)s?\b",
    re.IGNORECASE,
)
_NUMERIC_RESULT_RE = re.compile(
    r"""
    (?P<metric>[A-Za-z][A-Za-z0-9\-/+ ]{1,40})
    \s*(?:=|:)?\s*
    (?P<value>-?\d+(?:\.\d+)?)
    (?:\s*(?:-|–|to)\s*(?P<value_max>-?\d+(?:\.\d+)?))?
    (?P<unit>%|\s*%|\s*pts|\s*pp)?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_DATASET_RE = re.compile(r"on\s+(?P<dataset>[A-Za-z0-9\-_/\. ]{2,80})", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\b(test|dev|validation|val|train)\b", re.IGNORECASE)
_TASK_RE = re.compile(r"for\s+(?P<task>[A-Za-z][A-Za-z0-9\- ]{2,80})", re.IGNORECASE)
_CI_RE = re.compile(r"(?:\\u00B1\\s*\\d+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?\\s*-\\s*\\d+(?:\\.\\d+)?)")


def _normalize_graph_text(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _collect_sentence_indices_from_candidate(candidate: dict[str, Any]) -> list[int]:
    matches = candidate.get("evidence_spans") or []
    indices: set[int] = set()
    for match in matches:
        if not isinstance(match, dict):
            continue
        index = match.get("sentence_index")
        if isinstance(index, int):
            indices.add(index)
    return sorted(indices)


@dataclass
class NumericMeasurement:
    metric: str
    value: float
    value_text: str
    unit: Optional[str] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    task: Optional[str] = None
    ci: Optional[str] = None
    normalization_hash: Optional[str] = None
    value_range: Optional[tuple[float, float]] = None

    def compute_hash(self) -> str:
        parts = [
            _normalize_identifier(self.metric),
            f"{self.value:.6f}",
            (self.unit or "").strip().lower(),
            _normalize_identifier(self.dataset),
            _normalize_identifier(self.split),
            _normalize_identifier(self.task),
        ]
        if self.value_range:
            parts.append(f"{self.value_range[0]:.6f}")
            parts.append(f"{self.value_range[1]:.6f}")
        joined = "|".join(parts)
        self.normalization_hash = hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()
        return self.normalization_hash

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metric": self.metric,
            "value": self.value,
            "value_text": self.value_text,
        }
        if self.unit:
            payload["unit"] = self.unit
        if self.dataset:
            payload["dataset"] = self.dataset
        if self.split:
            payload["split"] = self.split
        if self.task:
            payload["task"] = self.task
        if self.ci:
            payload["confidence_interval"] = self.ci
        if self.normalization_hash:
            payload["normalization_hash"] = self.normalization_hash
        if self.value_range:
            payload["value_range"] = {
                "minimum": self.value_range[0],
                "maximum": self.value_range[1],
            }
        return payload


@dataclass
class CandidateWrapper:
    original_index: int
    order: tuple[int, int, int]
    data: dict[str, Any]
    section_id: Optional[str]
    sentence_index: Optional[int]
    measurement: Optional[NumericMeasurement] = None
    coref_resolved: bool = False
    table_match: bool = False


async def run_tier3_verifier(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]],
    enable_relation_fallback: Optional[bool] = None,
) -> dict[str, Any]:
    if base_summary is None:
        raise ValueError("Tier-3 verifier requires Tier-1 and Tier-2 summaries")

    summary = copy.deepcopy(base_summary)
    metadata = summary.setdefault("metadata", {})
    candidates_raw = list(summary.get("triple_candidates") or [])
    sections_raw = summary.get("sections") or []
    tables_raw = summary.get("tables") or []

    tiers = set(summary.get("tiers") or [])
    tiers.update({1, 2, 3})
    summary["tiers"] = sorted(tiers)

    fallback_candidates: list[dict[str, Any]] = []
    fallback_meta: dict[str, Any] = {}
    try:
        fallback_candidates, fallback_meta = await maybe_apply_relation_llm_fallback(
            paper_id,
            summary,
            enabled=enable_relation_fallback,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        fallback_meta = {
            "triggered": True,
            "status": "error",
            "errors": [str(exc)],
        }

    if fallback_candidates:
        candidates_raw.extend(fallback_candidates)
    if fallback_meta:
        metadata["tier3_fallback"] = fallback_meta

    if not candidates_raw:
        metadata["tier3"] = {
            "tier": TIER_NAME,
            "triple_count": 0,
            "normalized": 0,
            "table_matches": 0,
            "coref_resolved": 0,
        }
        return summary

    section_index = _build_section_index(sections_raw)
    table_index = _build_table_index(tables_raw)

    wrapped = _prepare_candidates(candidates_raw, section_index)
    _resolve_coreferences(wrapped)
    rejection_counts: dict[str, int] = {}
    _attach_measurements(wrapped, section_index, rejection_counts)
    _detect_table_matches(wrapped, table_index)
    _update_confidences(wrapped)

    summary["triple_candidates"] = [
        wrapper.data for wrapper in sorted(wrapped, key=lambda item: item.original_index)
    ]

    persisted_counts = await _persist_structured_results(paper_id, wrapped)

    tier3_meta = {
        "tier": TIER_NAME,
        "triple_count": len(wrapped),
        "normalized": sum(1 for wrapper in wrapped if wrapper.measurement),
        "table_matches": sum(1 for wrapper in wrapped if wrapper.table_match),
        "coref_resolved": sum(1 for wrapper in wrapped if wrapper.coref_resolved),
    }
    if rejection_counts:
        tier3_meta["numeric_rejections"] = rejection_counts
    if persisted_counts.get("results"):
        tier3_meta["persisted_results"] = persisted_counts["results"]
    if persisted_counts.get("relations"):
        tier3_meta["persisted_relations"] = persisted_counts["relations"]
    metadata["tier3"] = tier3_meta
    return summary


def _prepare_candidates(
    candidates: Sequence[dict[str, Any]],
    section_index: dict[str, dict[str, Any]],
) -> list[CandidateWrapper]:
    wrapped: list[CandidateWrapper] = []
    for idx, candidate in enumerate(candidates):
        section_id, sentence_index = _primary_span(candidate)
        section_meta = section_index.get(section_id or "", {})
        section_order = section_meta.get("order", 1_000_000)
        sentence_order = sentence_index if isinstance(sentence_index, int) else 1_000_000
        wrapper = CandidateWrapper(
            original_index=idx,
            order=(section_order, sentence_order, idx),
            data=copy.deepcopy(candidate),
            section_id=section_id,
            sentence_index=sentence_index if isinstance(sentence_index, int) else None,
        )
        wrapped.append(wrapper)
    wrapped.sort(key=lambda item: item.order)
    return wrapped


def _build_section_index(sections: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for order, section in enumerate(sections):
        section_id = str(section.get("section_id") or "").strip()
        if not section_id:
            continue
        index[section_id] = {
            "order": order,
            "sentences": section.get("sentence_spans") or [],
        }
    return index


def _build_table_index(tables: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        section_id = table.get("section_id")
        if not section_id:
            continue
        index.setdefault(str(section_id), []).append(table)
    return index


def _primary_span(candidate: dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
    spans = candidate.get("evidence_spans") or []
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            section_id = span.get("section_id")
            if section_id:
                sentence_index = span.get("sentence_index")
                return str(section_id), sentence_index if isinstance(sentence_index, int) else None
    return None, None


def _resolve_coreferences(candidates: Sequence[CandidateWrapper]) -> None:
    history: dict[str, dict[str, list[str]]] = {}

    for wrapper in candidates:
        candidate = wrapper.data
        section_key = wrapper.section_id or ""
        section_history = history.setdefault(section_key, {})

        subject = str(candidate.get("subject") or "").strip()
        subject_type = str(candidate.get("subject_type_guess") or "unknown").lower()
        resolved_subject, subject_alias = _resolve_entity(subject, subject_type, section_history)
        candidate["subject"] = resolved_subject
        if subject_alias:
            candidate["subject_alias"] = subject_alias
            wrapper.coref_resolved = True
        if not _needs_coref_resolution(resolved_subject):
            section_history.setdefault(subject_type, []).append(resolved_subject)

        obj = str(candidate.get("object") or "").strip()
        object_type = str(candidate.get("object_type_guess") or "unknown").lower()
        resolved_object, object_alias = _resolve_entity(obj, object_type, section_history)
        candidate["object"] = resolved_object
        if object_alias:
            candidate["object_alias"] = object_alias
            wrapper.coref_resolved = True
        if not _needs_coref_resolution(resolved_object):
            section_history.setdefault(object_type, []).append(resolved_object)


def _resolve_entity(
    text: str,
    type_guess: str,
    section_history: dict[str, list[str]],
) -> tuple[str, Optional[str]]:
    if not _needs_coref_resolution(text):
        return text, None

    candidates = section_history.get(type_guess) or []
    if candidates:
        return candidates[-1], text

    # Fallback to any antecedent within the section
    for records in section_history.values():
        if records:
            return records[-1], text
    return text, None


def _needs_coref_resolution(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if lowered in {"it", "they", "this", "that", "these", "those"}:
        return True
    if lowered.startswith("our ") or lowered.startswith("their "):
        return True
    if _COREF_HEAD_PATTERN.match(lowered) and _COREF_NOUN_PATTERN.search(lowered):
        return True
    return False


def _attach_measurements(
    candidates: Sequence[CandidateWrapper],
    section_index: dict[str, dict[str, Any]],
    rejection_counts: dict[str, int],
) -> None:
    for wrapper in candidates:
        measurement, rejection, verification_info = _extract_numeric_measure(
            wrapper.data, section_index
        )
        wrapper.measurement = measurement
        verification = wrapper.data.setdefault("verification", {})
        if verification_info:
            verification.update(verification_info)
        if measurement:
            measurement.compute_hash()
            payload = NumericMeasurementPayload.model_validate(measurement.to_payload())
            wrapper.data["normalization"] = payload.model_dump()
            verification["normalization_hash"] = measurement.normalization_hash
        elif rejection:
            rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1
        _update_graph_metadata(wrapper)


def _update_graph_metadata(wrapper: CandidateWrapper) -> None:
    metadata = wrapper.data.get("graph_metadata")
    if not isinstance(metadata, dict):
        return

    entities = metadata.setdefault("entities", {})
    pairs = metadata.setdefault("pairs", [])
    method_entry = entities.get("method")
    if not isinstance(method_entry, dict):
        return

    measurement = wrapper.measurement
    if measurement is None:
        return

    relation_guess = (wrapper.data.get("relation_type_guess") or "").strip().upper()
    dataset_allowed = relation_guess in _GRAPH_DATASET_RELATIONS or bool(measurement.dataset)
    metric_allowed = relation_guess in _GRAPH_METRIC_RELATIONS or bool(measurement.metric)
    task_allowed = relation_guess in _GRAPH_TASK_RELATIONS or bool(measurement.task)

    def _ensure_entry(entity_type: str, text: Optional[str]) -> Optional[dict[str, str]]:
        if not text:
            return None
        normalized = _normalize_graph_text(text)
        if not normalized:
            return None
        lowered = normalized.lower()
        existing = entities.get(entity_type)
        if isinstance(existing, dict) and existing.get("normalized") == lowered:
            return existing
        entry = {"text": text.strip(), "normalized": lowered}
        entities[entity_type] = entry
        return entry

    def _has_pair(target_type: str, normalized: str) -> bool:
        for pair in pairs:
            target = pair.get("target")
            if not isinstance(target, dict):
                continue
            if target.get("type") == target_type and target.get("normalized") == normalized:
                return True
        return False

    sentence_indices = _collect_sentence_indices_from_candidate(wrapper.data)
    confidence = wrapper.data.get("triple_conf")
    evidence_text = wrapper.data.get("evidence")

    def _append_pair(
        target_type: str,
        text: Optional[str],
        relation: str,
        *,
        allowed: bool,
    ) -> None:
        if not allowed:
            return
        entry = _ensure_entry(target_type, text)
        if entry is None:
            return
        normalized = entry.get("normalized")
        if normalized and _has_pair(target_type, normalized):
            return
        pairs.append(
            {
                "source": {"type": "method", **method_entry},
                "target": {"type": target_type, **entry},
                "relation": relation,
                "confidence": confidence,
                "section_id": wrapper.section_id,
                "sentence_indices": sentence_indices,
                "evidence": evidence_text,
                "provenance": "tier3_measurement",
            }
        )

    _append_pair(
        "dataset",
        measurement.dataset,
        "evaluates_on",
        allowed=dataset_allowed,
    )
    _append_pair(
        "metric",
        measurement.metric,
        "reports",
        allowed=metric_allowed,
    )
    _append_pair(
        "task",
        measurement.task,
        "proposes",
        allowed=task_allowed,
    )


def _extract_numeric_measure(
    candidate: dict[str, Any],
    section_index: dict[str, dict[str, Any]],
) -> tuple[Optional[NumericMeasurement], Optional[str], dict[str, Any]]:
    texts: list[str] = []
    for key in ("object", "evidence"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)

    seen_metrics: set[str] = set()
    verification: dict[str, Any] = {}

    for text in texts:
        for match in _NUMERIC_RESULT_RE.finditer(text):
            metric = _clean_metric(match.group("metric"))
            if not metric or metric.lower() in seen_metrics:
                continue
            seen_metrics.add(metric.lower())

            value_raw = match.group("value")
            value_max_raw = match.group("value_max")
            try:
                value_numeric = float(value_raw)
            except (TypeError, ValueError):
                continue

            unit = _clean_unit(match.group("unit"))
            dataset = _extract_dataset(text)
            split = _extract_split(text)
            task = _extract_task(text)
            ci = _extract_ci(text)

            measurement = NumericMeasurement(
                metric=metric,
                value=value_numeric,
                value_text=value_raw,
                unit=unit,
                dataset=dataset,
                split=split,
                task=task,
                ci=ci,
            )

            if value_max_raw:
                try:
                    value_max = float(value_max_raw)
                except ValueError:
                    value_max = None
                else:
                    low, high = sorted((value_numeric, value_max))
                    measurement.value_range = (low, high)
                    measurement.value_text = f"{value_raw}-{value_max_raw}"

            aliases = _value_aliases(measurement)
            verification["numeric_aliases"] = sorted(set(aliases))
            match_found, matched_alias = _value_in_evidence(
                candidate, measurement, section_index, aliases
            )
            verification["value_in_evidence"] = match_found
            if matched_alias:
                verification["matched_alias"] = matched_alias

            if not match_found:
                verification["numeric_status"] = "rejected"
                verification["numeric_reason"] = "value_not_in_evidence"
                return None, "value_not_in_evidence", verification

            verification["numeric_status"] = "accepted"
            verification["numeric_reason"] = "value_in_evidence"
            verification["numeric_value"] = value_numeric
            return measurement, None, verification

    if texts:
        verification.setdefault("numeric_status", "skipped")
    return None, None, verification


def _clean_metric(metric: str) -> str:
    cleaned = metric.strip(" :")
    cleaned = re.sub(r"\b(score|value)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _clean_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    cleaned = unit.strip().replace(" ", "")
    if not cleaned:
        return None
    replacements = {"pts": "points", "%": "%", "pp": "pp"}
    return replacements.get(cleaned.lower(), cleaned)


def _extract_dataset(text: str) -> Optional[str]:
    match = _DATASET_RE.search(text)
    if not match:
        return None
    dataset = match.group("dataset").strip()
    dataset = re.split(r"\b(test|dev|validation|dataset|set|split)\b", dataset, maxsplit=1)[0].strip()
    dataset = dataset.strip(" ,.;:)")
    if len(dataset) < 2:
        return None
    return dataset


def _extract_split(text: str) -> Optional[str]:
    match = _SPLIT_RE.search(text)
    if not match:
        return None
    token = match.group(1).lower()
    mapping = {"val": "validation", "dev": "dev"}
    return mapping.get(token, token)


def _extract_task(text: str) -> Optional[str]:
    match = _TASK_RE.search(text)
    if not match:
        return None
    task = match.group("task").strip()
    task = re.split(r"\b(on|using|with)\b", task, maxsplit=1)[0].strip()
    task = task.strip(" ,.;:)")
    if len(task) < 3:
        return None
    return task


def _extract_ci(text: str) -> Optional[str]:
    match = _CI_RE.search(text)
    if not match:
        return None
    return match.group(0).strip()


def _detect_table_matches(
    candidates: Sequence[CandidateWrapper],
    table_index: dict[str, list[dict[str, Any]]],
) -> None:
    for wrapper in candidates:
        measurement = wrapper.measurement
        if not measurement or not wrapper.section_id:
            continue
        tables = table_index.get(wrapper.section_id)
        if not tables:
            continue
        aliases = _value_aliases(measurement)
        match_info: Optional[dict[str, Any]] = None
        for table in tables:
            for cell in table.get("cells", []):
                cell_text = str(cell.get("text") or "")
                if not cell_text:
                    continue
                lowered = cell_text.lower()
                if any(alias.lower() in lowered for alias in aliases):
                    if measurement.metric and measurement.metric.lower() not in lowered:
                        # Allow matches even without metric mention but prefer those that do
                        pass
                    match_info = {
                        "table_id": table.get("table_id"),
                        "cell": cell,
                    }
                    break
            if match_info:
                break
        verification = wrapper.data.setdefault("verification", {})
        if match_info:
            wrapper.table_match = True
            verification["table_match"] = match_info
        else:
            verification.setdefault("table_match", False)


def _collect_evidence_texts(
    candidate: dict[str, Any],
    section_index: dict[str, dict[str, Any]],
) -> list[str]:
    texts: list[str] = []
    evidence_snippet = candidate.get("evidence")
    if isinstance(evidence_snippet, str) and evidence_snippet.strip():
        texts.append(evidence_snippet)
    spans = candidate.get("evidence_spans") or []
    for span in spans:
        if not isinstance(span, dict):
            continue
        section_id = span.get("section_id")
        if not section_id:
            continue
        sentence_index = span.get("sentence_index")
        if not isinstance(sentence_index, int):
            continue
        section_meta = section_index.get(str(section_id)) or {}
        sentences = section_meta.get("sentences") or []
        if 0 <= sentence_index < len(sentences):
            sentence_entry = sentences[sentence_index]
            if isinstance(sentence_entry, dict):
                sentence_text = sentence_entry.get("text")
            else:
                sentence_text = sentence_entry
            if isinstance(sentence_text, str) and sentence_text.strip():
                texts.append(sentence_text)

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for text in texts:
        normalized = text.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _alias_pattern(alias: str) -> Optional[re.Pattern[str]]:
    cleaned = alias.strip().lower()
    if not cleaned:
        return None
    try:
        return re.compile(rf"(?<![\d.]){re.escape(cleaned)}(?![\d.])")
    except re.error:
        return None


def _value_in_evidence(
    candidate: dict[str, Any],
    measurement: NumericMeasurement,
    section_index: dict[str, dict[str, Any]],
    aliases: Sequence[str],
) -> tuple[bool, Optional[str]]:
    texts = _collect_evidence_texts(candidate, section_index)
    if not texts or not aliases:
        return False, None

    lowered_texts = [text.lower() for text in texts]
    for alias in aliases:
        pattern = _alias_pattern(alias)
        if pattern is None:
            continue
        for text in lowered_texts:
            if pattern.search(text):
                return True, alias
    return False, None


def _value_aliases(measurement: NumericMeasurement) -> list[str]:
    aliases: set[str] = set()

    def _format_value(value: float) -> str:
        formatted = f"{value:.4f}".rstrip("0").rstrip(".")
        return formatted or str(value)

    if measurement.value_text:
        aliases.add(measurement.value_text.strip())

    aliases.add(_format_value(measurement.value))
    if float(int(measurement.value)) == measurement.value:
        aliases.add(str(int(measurement.value)))

    if measurement.value_range:
        low, high = measurement.value_range
        aliases.add(_format_value(low))
        aliases.add(_format_value(high))
        if float(int(low)) == low:
            aliases.add(str(int(low)))
        if float(int(high)) == high:
            aliases.add(str(int(high)))
        range_dash = f"{_format_value(low)}-{_format_value(high)}"
        range_en_dash = range_dash.replace("-", "–")
        aliases.add(range_dash)
        aliases.add(range_en_dash)

    expanded_aliases: set[str] = set()
    for alias in list(aliases):
        cleaned = alias.strip()
        if not cleaned:
            continue
        expanded_aliases.add(cleaned)
        if "." in cleaned:
            expanded_aliases.add(cleaned.replace(".", ","))

    aliases = expanded_aliases

    unit = measurement.unit or ""
    if unit and unit.lower() not in {"points", ""}:
        for alias in list(aliases):
            aliases.add(f"{alias}{unit}")
            aliases.add(f"{alias} {unit}")

    return [alias for alias in aliases if alias]


def _update_confidences(candidates: Sequence[CandidateWrapper]) -> None:
    normalization_counts = _count_normalizations(candidates)
    duplicate_counts = _count_duplicate_text(candidates)

    for wrapper in candidates:
        data = wrapper.data
        base_conf = data.get("triple_conf")
        try:
            score = float(base_conf)
        except (TypeError, ValueError):
            score = _BASE_CONFIDENCE_FALLBACK
        score = max(0.0, min(1.0, score))

        components: dict[str, float] = {"tier2_base": round(score, 4)}

        if wrapper.measurement is not None:
            score = _apply_adjustment(score, components, "json_valid", _JSON_VALID_BONUS)

        if wrapper.coref_resolved:
            score = _apply_adjustment(score, components, "coref_resolution", _COREF_BONUS)
        if wrapper.table_match:
            score = _apply_adjustment(score, components, "table_match", _TABLE_MATCH_BONUS)

        provenance_category = _provenance_category(wrapper.data, wrapper)
        prior_adjustment = _PROVENANCE_PRIORS.get(provenance_category)
        if prior_adjustment:
            score = _apply_adjustment(
                score, components, f"provenance_{provenance_category}", prior_adjustment
            )
        verification = data.setdefault("verification", {})
        verification.setdefault("provenance_category", provenance_category)

        if wrapper.measurement and wrapper.measurement.normalization_hash:
            occurrences = normalization_counts.get(wrapper.measurement.normalization_hash, 0)
            if occurrences > 1:
                bonus = min(_DUPLICATE_BONUS_STEP * (occurrences - 1), _DUPLICATE_BONUS_CAP)
                score = _apply_adjustment(score, components, "duplicate_support", bonus)
        else:
            signature = _triple_signature(data)
            occurrences = duplicate_counts.get(signature, 0)
            if occurrences > 1:
                bonus = min(0.02 * (occurrences - 1), 0.06)
                score = _apply_adjustment(score, components, "duplicate_support", bonus)

        final_score = round(min(1.0, score), 4)
        components["final"] = final_score
        data["triple_conf"] = final_score
        data["confidence_components"] = components
        verification.setdefault("coref_resolved", wrapper.coref_resolved)
        verification.setdefault("table_match", wrapper.table_match)



async def _persist_structured_results(
    paper_id: UUID,
    candidates: Sequence[CandidateWrapper],
) -> dict[str, int]:
    results: list[ResultCreate] = []
    result_keys: set[tuple[Any, ...]] = set()
    relations: list[MethodRelationCreate] = []
    relation_keys: set[tuple[Any, ...]] = set()

    for wrapper in candidates:
        result_outcome, relation_outcomes = await _candidate_to_result(paper_id, wrapper)
        if result_outcome:
            result, key = result_outcome
            if key not in result_keys:
                result_keys.add(key)
                results.append(result)
        for relation, key in relation_outcomes:
            if key in relation_keys:
                continue
            relation_keys.add(key)
            relations.append(relation)

    inserted_results = await append_results(paper_id, results) if results else []
    inserted_relations = (
        await append_method_relations(paper_id, relations) if relations else []
    )

    def _persisted_count(returned: Any, attempted: Sequence[Any]) -> int:
        if not attempted:
            return 0
        if returned is None:
            return len(attempted)
        try:
            count = len(returned)  # type: ignore[arg-type]
        except TypeError:
            return len(attempted)
        return count or len(attempted)

    return {
        "results": _persisted_count(inserted_results, results),
        "relations": _persisted_count(inserted_relations, relations),
    }


async def _candidate_to_result(
    paper_id: UUID,
    wrapper: CandidateWrapper,
) -> tuple[
    Optional[tuple[ResultCreate, tuple[Any, ...]]],
    list[tuple[MethodRelationCreate, tuple[Any, ...]]],
]:
    data = wrapper.data
    subject = (data.get("subject") or "").strip()
    obj = (data.get("object") or "").strip()
    subject_type = (data.get("subject_type_guess") or "unknown").lower()
    object_type = (data.get("object_type_guess") or "unknown").lower()
    relation_guess = (data.get("relation_type_guess") or "other").upper()
    measurement = wrapper.measurement

    method_name = _clean_entity_name(subject) if subject_type == "method" else None
    if not method_name and object_type == "method":
        method_name = _clean_entity_name(obj)
    if not method_name and relation_guess in {"MEASURES", "REPORTS", "ACHIEVES", "EVALUATED_ON", "USES", "OUTPERFORMS", "COMPARED_TO"}:
        method_name = _clean_entity_name(subject)

    dataset_name = None
    if measurement and measurement.dataset:
        dataset_name = _clean_entity_name(measurement.dataset)
    if not dataset_name and object_type == "dataset":
        dataset_name = _clean_entity_name(obj)
    if not dataset_name and relation_guess in {"EVALUATED_ON", "USES"}:
        dataset_name = _clean_entity_name(obj)

    metric_name = None
    metric_unit = None
    if measurement and measurement.metric:
        metric_name = _clean_entity_name(measurement.metric)
        metric_unit = measurement.unit
    if not metric_name and data.get("metric_inference"):
        metric_name = _clean_entity_name(data["metric_inference"].get("normalized_metric"))
    if not metric_name and object_type == "metric":
        metric_name = _clean_entity_name(obj)

    task_name = None
    if measurement and measurement.task:
        task_name = _clean_entity_name(measurement.task)
    if not task_name and object_type == "task":
        task_name = _clean_entity_name(obj)
    if not task_name and subject_type == "task":
        task_name = _clean_entity_name(subject)
    if not task_name and relation_guess == "PROPOSES":
        task_name = _clean_entity_name(obj)

    if subject_type == "task" and not method_name and object_type == "method":
        method_name = _clean_entity_name(obj)

    if not method_name:
        return None, []

    if not any([dataset_name, metric_name, task_name, measurement]):
        return None, []

    method_model = await ensure_method(method_name)
    dataset_model = await ensure_dataset(dataset_name) if dataset_name else None
    metric_model = await ensure_metric(metric_name, unit=metric_unit) if metric_name else None
    task_model = await ensure_task(task_name) if task_name else None

    value_numeric = None
    value_text: Optional[str] = None
    if measurement:
        value_text = measurement.value_text or obj or None
        if measurement.value is not None:
            try:
                value_numeric = Decimal(str(measurement.value))
            except (TypeError, ValueError):
                value_numeric = None
    if value_text is None:
        value_text = obj or None

    evidence: list[dict[str, Any]] = []
    snippet = data.get("evidence")
    if snippet:
        evidence_item = {
            "snippet": snippet,
            "section_id": wrapper.section_id,
            "candidate_id": data.get("candidate_id"),
            "relation": data.get("relation"),
            "tier": data.get("tier"),
        }
        evidence.append(evidence_item)

    confidence = data.get("triple_conf")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    valid_spans: list[tuple[str, int]] = []
    raw_spans = data.get("evidence_spans") or []
    if isinstance(raw_spans, list):
        for span in raw_spans:
            if not isinstance(span, dict):
                continue
            section_id = span.get("section_id")
            sentence_index = span.get("sentence_index")
            if section_id and isinstance(sentence_index, int):
                valid_spans.append((str(section_id), sentence_index))

    has_verified_span = bool(valid_spans)
    distinct_span_positions = {span for span in valid_spans}
    multi_verified_spans = len(distinct_span_positions) > 1

    snippet_text = str(snippet or "")
    snippet_lower = snippet_text.lower()

    result_outcome: Optional[tuple[ResultCreate, tuple[Any, ...]]] = None
    relation_outcomes: list[tuple[MethodRelationCreate, tuple[Any, ...]]] = []

    if measurement:
        verification_info = data.get("verification") or {}
        numeric_status = verification_info.get("numeric_status")
        numeric_reason = verification_info.get("numeric_reason")
        verified_flag: Optional[bool] = None
        if numeric_status == "accepted":
            verified_flag = True
        elif numeric_status == "rejected":
            verified_flag = False
        verifier_notes = None
        if numeric_reason:
            note_parts = [f"tier3:{numeric_reason}"]
            matched_alias = verification_info.get("matched_alias")
            if matched_alias:
                note_parts.append(f"alias={matched_alias}")
            verifier_notes = "; ".join(note_parts)

        result = ResultCreate(
            paper_id=paper_id,
            method_id=method_model.id,
            dataset_id=dataset_model.id if dataset_model else None,
            metric_id=metric_model.id if metric_model else None,
            task_id=task_model.id if task_model else None,
            split=measurement.split,
            value_numeric=value_numeric,
            value_text=value_text,
            is_sota=False,
            confidence=confidence_value,
            evidence=evidence,
            verified=verified_flag,
            verifier_notes=verifier_notes,
        )

        key = (
            result.method_id,
            result.dataset_id,
            result.metric_id,
            result.task_id,
            result.split,
            (result.value_text or "").strip(),
            str(result.value_numeric) if result.value_numeric is not None else None,
        )
        result_outcome = (result, key)
    else:
        if confidence_value is None:
            confidence_value = 0.0
        dataset_signals = 0
        if relation_guess in _DATASET_RELATION_GUESSES:
            dataset_signals += 1
        if object_type == "dataset":
            dataset_signals += 1
        if dataset_name and dataset_name.lower() in snippet_lower:
            dataset_signals += 1
        elif snippet_text and _DATASET_RE.search(snippet_text):
            dataset_signals += 1
        if multi_verified_spans:
            dataset_signals += 1

        task_signals = 0
        if relation_guess in _TASK_RELATION_GUESSES:
            task_signals += 1
        if object_type == "task" or subject_type == "task":
            task_signals += 1
        if task_name and task_name.lower() in snippet_lower:
            task_signals += 1
        elif snippet_text and _TASK_RE.search(snippet_text):
            task_signals += 1
        if multi_verified_spans:
            task_signals += 1

        dataset_threshold = _compute_qualitative_threshold(
            _QUALITATIVE_RELATION_CONFIDENCE,
            has_verified_span=has_verified_span,
            signal_strength=dataset_signals,
            type_guard=subject_type == "method" and object_type == "dataset",
        )
        task_threshold = _compute_qualitative_threshold(
            _QUALITATIVE_RELATION_CONFIDENCE,
            has_verified_span=has_verified_span,
            signal_strength=task_signals,
            type_guard=(
                (subject_type == "method" and object_type == "task")
                or (subject_type == "task" and object_type == "method")
            ),
        )

        if dataset_model and confidence_value >= dataset_threshold:
            relation = MethodRelationCreate(
                paper_id=paper_id,
                method_id=method_model.id,
                dataset_id=dataset_model.id,
                task_id=None,
                relation_type=MethodRelationType.EVALUATES_ON,
                confidence=confidence_value,
                evidence=evidence,
            )
            key = (
                relation.method_id,
                relation.dataset_id,
                relation.task_id,
                relation.relation_type,
            )
            relation_outcomes.append((relation, key))
        if task_model and confidence_value >= task_threshold:
            relation = MethodRelationCreate(
                paper_id=paper_id,
                method_id=method_model.id,
                dataset_id=None,
                task_id=task_model.id,
                relation_type=MethodRelationType.PROPOSES,
                confidence=confidence_value,
                evidence=evidence,
            )
            key = (
                relation.method_id,
                relation.dataset_id,
                relation.task_id,
                relation.relation_type,
            )
            relation_outcomes.append((relation, key))

    return result_outcome, relation_outcomes


def _compute_qualitative_threshold(
    base_threshold: float,
    *,
    has_verified_span: bool,
    signal_strength: int,
    type_guard: bool,
) -> float:
    if not has_verified_span or not type_guard:
        return base_threshold
    if signal_strength < 2:
        return base_threshold
    reduction = 0.05 * min(signal_strength, 4)
    return max(0.5, base_threshold - reduction)


def _clean_entity_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _apply_adjustment(
    score: float, components: dict[str, float], key: str, delta: float
) -> float:
    if not delta:
        return score
    components[key] = round(delta, 4)
    updated = score + delta
    return max(0.0, min(1.0, updated))


def _provenance_category(
    candidate: dict[str, Any], wrapper: CandidateWrapper
) -> str:
    if wrapper.table_match:
        return "table"

    provenance_value = candidate.get("provenance")
    tokens: list[str] = []

    if isinstance(provenance_value, str):
        tokens.append(provenance_value)
    elif isinstance(provenance_value, dict):
        for key in ("source", "tier", "type", "name", "provider"):
            value = provenance_value.get(key)
            if isinstance(value, str):
                tokens.append(value)

    tier_value = candidate.get("tier")
    if isinstance(tier_value, str):
        tokens.append(tier_value)

    source_value = candidate.get("source")
    if isinstance(source_value, str):
        tokens.append(source_value)

    for token in tokens:
        lowered = token.lower()
        if "table" in lowered:
            return "table"
        if "rule" in lowered:
            return "rule"
        if "spacy" in lowered or "ner" in lowered:
            return "spacy"
        if "llm" in lowered or "gpt" in lowered:
            return "llm"

    return "unknown"


def _count_normalizations(candidates: Sequence[CandidateWrapper]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for wrapper in candidates:
        if wrapper.measurement and wrapper.measurement.normalization_hash:
            key = wrapper.measurement.normalization_hash
            counts[key] = counts.get(key, 0) + 1
    return counts


def _count_duplicate_text(candidates: Sequence[CandidateWrapper]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for wrapper in candidates:
        signature = _triple_signature(wrapper.data)
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def _triple_signature(candidate: dict[str, Any]) -> str:
    parts = [
        _normalize_identifier(candidate.get("subject")),
        _normalize_identifier(candidate.get("relation")),
        _normalize_identifier(candidate.get("object")),
    ]
    return "|".join(parts)


def _normalize_identifier(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = re.sub(r"\s+", " ", str(value)).strip().lower()
    return normalized
