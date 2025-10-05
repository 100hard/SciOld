from __future__ import annotations

from uuid import uuid4

import pytest
from types import SimpleNamespace
from typing import Optional

from app.services.extraction_tier3 import run_tier3_verifier, TIER_NAME


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_section(section_id: str, sentences: list[str]) -> dict[str, object]:
    return {
        "section_id": section_id,
        "section_hash": f"hash-{section_id}",
        "page_number": 1,
        "char_start": 0,
        "char_end": 100,
        "sentence_spans": [
            {"start": idx * 10, "end": idx * 10 + len(text), "text": text}
            for idx, text in enumerate(sentences)
        ],
    }


@pytest.mark.anyio
async def test_run_tier3_verifier_resolves_coref_and_updates_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    sections = [
        _build_section(
            "sec-1",
            [
                "AlphaNet is a new translation model.",
                "AlphaNet achieves BLEU 41.8 on WMT14 En-Fr test set.",
                "This model further improves BLEU 41.8 on the same benchmark.",
            ],
        )
    ]

    tables = [
        {
            "table_id": "table_1_1",
            "section_id": "sec-1",
            "page_number": 1,
            "cells": [
                {"row": 0, "column": 0, "text": "Model"},
                {"row": 0, "column": 1, "text": "BLEU 41.8"},
            ],
        }
    ]

    triple_candidates = [
        {
            "candidate_id": "tier2_llm_openie_001",
            "tier": "tier2_llm_openie",
            "subject": "AlphaNet",
            "relation": "achieves",
            "object": "BLEU 41.8 on WMT14 En-Fr test set",
            "subject_type_guess": "method",
            "relation_type_guess": "achieves",
            "object_type_guess": "result",
            "triple_conf": 0.6,
            "evidence": "AlphaNet achieves BLEU 41.8 on WMT14 En-Fr test set.",
            "evidence_spans": [
                {"section_id": "sec-1", "sentence_index": 1, "start": 0, "end": 52}
            ],
        },
        {
            "candidate_id": "tier2_llm_openie_002",
            "tier": "tier2_llm_openie",
            "subject": "This model",
            "relation": "achieves",
            "object": "BLEU 41.8 on WMT14 En-Fr test set",
            "subject_type_guess": "method",
            "relation_type_guess": "achieves",
            "object_type_guess": "result",
            "triple_conf": 0.58,
            "evidence": "This model further improves BLEU 41.8 on the same benchmark.",
            "evidence_spans": [
                {"section_id": "sec-1", "sentence_index": 2, "start": 0, "end": 64}
            ],
        },
    ]

    captured_results: list = []
    method_cache: dict[str, SimpleNamespace] = {}
    dataset_cache: dict[str, SimpleNamespace] = {}
    metric_cache: dict[str, SimpleNamespace] = {}
    task_cache: dict[str, SimpleNamespace] = {}

    async def _ensure_cached(cache: dict[str, SimpleNamespace], name: str) -> SimpleNamespace:
        key = (name or "").strip()
        if not key:
            raise ValueError("Name cannot be empty")
        existing = cache.get(key)
        if existing is None:
            existing = SimpleNamespace(id=uuid4(), name=key, aliases=[], description=None, unit=None)
            cache[key] = existing
        return existing

    async def fake_ensure_method(name: str, **_: object) -> SimpleNamespace:
        return await _ensure_cached(method_cache, name)

    async def fake_ensure_dataset(name: str, **_: object) -> SimpleNamespace:
        return await _ensure_cached(dataset_cache, name)

    async def fake_ensure_metric(name: str, *, unit: Optional[str] = None, **_: object) -> SimpleNamespace:
        model = await _ensure_cached(metric_cache, name)
        model.unit = unit
        return model

    async def fake_ensure_task(name: str, **_: object) -> SimpleNamespace:
        return await _ensure_cached(task_cache, name)

    async def fake_append_results(paper_id_arg, results):
        if paper_id_arg != paper_id:
            raise AssertionError("Unexpected paper_id")
        captured_results.extend(results)
        return []

    monkeypatch.setattr("app.services.extraction_tier3.ensure_method", fake_ensure_method)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_metric", fake_ensure_metric)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_task", fake_ensure_task)
    monkeypatch.setattr("app.services.extraction_tier3.append_results", fake_append_results)
    async def fake_append_method_relations(*_: object, **__: object) -> list:
        return []

    monkeypatch.setattr(
        "app.services.extraction_tier3.append_method_relations",
        fake_append_method_relations,
    )

    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1, 2],
        "sections": sections,
        "tables": tables,
        "triple_candidates": triple_candidates,
    }

    summary = await run_tier3_verifier(paper_id, base_summary=base_summary)

    assert captured_results, "Tier-3 should emit structured results"
    assert captured_results[0].verified is True
    assert captured_results[0].verifier_notes.startswith("tier3:value_in_evidence")
    assert 3 in summary["tiers"], "tier3 should be recorded"

    candidates = summary["triple_candidates"]
    first, second = candidates

    assert first["normalization"]["metric"].lower().startswith("bleu")
    assert first["verification"]["table_match"]
    assert first["verification"]["numeric_status"] == "accepted"
    assert first["verification"]["provenance_category"] == "table"

    assert second["subject"] == "AlphaNet"
    assert second["subject_alias"] == "This model"
    assert second["verification"]["coref_resolved"] is True

    assert second["triple_conf"] > 0.58
    assert "duplicate_support" in second["confidence_components"]

    meta = summary["metadata"]["tier3"]
    assert meta["tier"] == TIER_NAME
    assert meta["coref_resolved"] >= 1
    assert meta["table_matches"] >= 1
    assert "numeric_rejections" not in meta


@pytest.mark.anyio
async def test_run_tier3_verifier_rejects_measurement_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    sections = [
        _build_section(
            "sec-1",
            [
                "AlphaNet is a new translation model.",
                "AlphaNet achieves state-of-the-art translation accuracy.",
            ],
        )
    ]

    triple_candidates = [
        {
            "candidate_id": "tier2_llm_openie_003",
            "tier": "tier2_llm_openie",
            "subject": "AlphaNet",
            "relation": "achieves",
            "object": "BLEU 41.8 on WMT14 En-Fr test set",
            "subject_type_guess": "method",
            "relation_type_guess": "achieves",
            "object_type_guess": "result",
            "triple_conf": 0.6,
            "evidence": "AlphaNet achieves state-of-the-art translation accuracy.",
            "evidence_spans": [
                {"section_id": "sec-1", "sentence_index": 1, "start": 0, "end": 62}
            ],
        }
    ]

    captured_results: list = []

    async def fake_append_results(paper_id_arg, results):
        if paper_id_arg != paper_id:
            raise AssertionError("Unexpected paper_id")
        captured_results.extend(results)
        return []

    async def fake_append_method_relations(*_: object, **__: object) -> list:
        return []

    async def fake_ensure_method(name: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(id=uuid4(), name=name)

    async def fake_ensure_dataset(*_: object, **__: object) -> None:
        return None

    async def fake_ensure_metric(*_: object, **__: object) -> None:
        return None

    async def fake_ensure_task(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr("app.services.extraction_tier3.ensure_method", fake_ensure_method)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_metric", fake_ensure_metric)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_task", fake_ensure_task)
    monkeypatch.setattr("app.services.extraction_tier3.append_results", fake_append_results)
    monkeypatch.setattr(
        "app.services.extraction_tier3.append_method_relations",
        fake_append_method_relations,
    )

    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1, 2],
        "sections": sections,
        "tables": [],
        "triple_candidates": triple_candidates,
    }

    summary = await run_tier3_verifier(paper_id, base_summary=base_summary)

    assert not captured_results, "Measurement without evidence should not persist"

    candidate = summary["triple_candidates"][0]
    assert "normalization" not in candidate
    assert candidate["verification"]["numeric_status"] == "rejected"
    assert candidate["verification"]["value_in_evidence"] is False

    meta = summary["metadata"]["tier3"]
    assert meta["numeric_rejections"]["value_not_in_evidence"] == 1


@pytest.mark.anyio
async def test_run_tier3_verifier_persists_qualitative_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    sections = [
        _build_section(
            "sec-1",
            [
                "QualNet is introduced for transfer learning.",
                "QualNet is evaluated on the GLUE benchmark to assess generalisation.",
            ],
        )
    ]

    triple_candidates = [
        {
            "candidate_id": "tier2_llm_openie_qual_001",
            "tier": "tier2_llm_openie",
            "subject": "QualNet",
            "relation": "evaluated on",
            "object": "GLUE benchmark",
            "subject_type_guess": "method",
            "relation_type_guess": "evaluated_on",
            "object_type_guess": "dataset",
            "triple_conf": 0.72,
            "evidence": "QualNet is evaluated on the GLUE benchmark to assess generalisation.",
            "evidence_spans": [
                {"section_id": "sec-1", "sentence_index": 1, "start": 0, "end": 74}
            ],
        }
    ]

    captured_results: list = []
    captured_relations: list = []
    method_cache: dict[str, SimpleNamespace] = {}
    dataset_cache: dict[str, SimpleNamespace] = {}

    async def _ensure_cached(cache: dict[str, SimpleNamespace], name: str) -> SimpleNamespace:
        key = (name or "").strip()
        if not key:
            raise ValueError("Name cannot be empty")
        existing = cache.get(key)
        if existing is None:
            existing = SimpleNamespace(id=uuid4(), name=key, aliases=[], description=None)
            cache[key] = existing
        return existing

    async def fake_ensure_method(name: str, **_: object) -> SimpleNamespace:
        return await _ensure_cached(method_cache, name)

    async def fake_ensure_dataset(name: str, **_: object) -> SimpleNamespace:
        return await _ensure_cached(dataset_cache, name)

    async def fake_append_results(paper_id_arg, results):
        if paper_id_arg != paper_id:
            raise AssertionError("Unexpected paper_id")
        captured_results.extend(results)
        return []

    async def fake_append_method_relations(paper_id_arg, relations):
        if paper_id_arg != paper_id:
            raise AssertionError("Unexpected paper_id")
        captured_relations.extend(relations)
        return []

    monkeypatch.setattr("app.services.extraction_tier3.ensure_method", fake_ensure_method)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_dataset", fake_ensure_dataset)
    async def fake_ensure_metric(*_: object, **__: object) -> None:
        return None

    async def fake_ensure_task(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr("app.services.extraction_tier3.ensure_metric", fake_ensure_metric)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_task", fake_ensure_task)
    monkeypatch.setattr("app.services.extraction_tier3.append_results", fake_append_results)
    monkeypatch.setattr(
        "app.services.extraction_tier3.append_method_relations",
        fake_append_method_relations,
    )

    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1, 2],
        "sections": sections,
        "tables": [],
        "triple_candidates": triple_candidates,
    }

    summary = await run_tier3_verifier(paper_id, base_summary=base_summary)

    assert not captured_results, "Qualitative relations should not persist as numeric results"
    assert captured_relations, "Qualitative relations should be captured"
    relation = captured_relations[0]
    assert relation.dataset_id is not None
    assert relation.relation_type.value == "evaluates_on"
    assert relation.confidence == pytest.approx(0.67, rel=1e-3)
    assert relation.evidence and relation.evidence[0]["snippet"].startswith("QualNet is evaluated")

    meta = summary["metadata"]["tier3"]
    assert meta.get("persisted_relations") == len(captured_relations)


@pytest.mark.anyio
async def test_run_tier3_verifier_allows_supported_low_confidence_dataset_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    sections = [
        _build_section(
            "sec-1",
            [
                "QualNet is introduced for transfer learning.",
                "QualNet is evaluated on the GLUE benchmark in our experiments.",
                "Results on the GLUE benchmark highlight broad coverage.",
            ],
        )
    ]

    triple_candidates = [
        {
            "candidate_id": "tier2_llm_openie_qual_002",
            "tier": "tier2_llm_openie",
            "subject": "QualNet",
            "relation": "evaluated on",
            "object": "GLUE benchmark",
            "subject_type_guess": "method",
            "relation_type_guess": "evaluated_on",
            "object_type_guess": "dataset",
            "triple_conf": 0.56,
            "evidence": "QualNet is evaluated on the GLUE benchmark in our experiments.",
            "evidence_spans": [
                {"section_id": "sec-1", "sentence_index": 1, "start": 0, "end": 63},
                {"section_id": "sec-1", "sentence_index": 2, "start": 0, "end": 59},
            ],
        }
    ]

    captured_relations: list = []
    method_cache: dict[str, SimpleNamespace] = {}
    dataset_cache: dict[str, SimpleNamespace] = {}

    async def _ensure_cached(cache: dict[str, SimpleNamespace], name: str) -> SimpleNamespace:
        key = (name or "").strip()
        if not key:
            raise ValueError("Name cannot be empty")
        existing = cache.get(key)
        if existing is None:
            existing = SimpleNamespace(id=uuid4(), name=key, aliases=[], description=None)
            cache[key] = existing
        return existing

    async def fake_ensure_method(name: str, **_: object) -> SimpleNamespace:
        return await _ensure_cached(method_cache, name)

    async def fake_ensure_dataset(name: str, **_: object) -> SimpleNamespace:
        return await _ensure_cached(dataset_cache, name)

    async def fake_append_results(*_: object, **__: object) -> list:
        return []

    async def fake_append_method_relations(paper_id_arg, relations):
        if paper_id_arg != paper_id:
            raise AssertionError("Unexpected paper_id")
        captured_relations.extend(relations)
        return []

    async def fake_ensure_metric(*_: object, **__: object) -> None:
        return None

    async def fake_ensure_task(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr("app.services.extraction_tier3.ensure_method", fake_ensure_method)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_metric", fake_ensure_metric)
    monkeypatch.setattr("app.services.extraction_tier3.ensure_task", fake_ensure_task)
    monkeypatch.setattr("app.services.extraction_tier3.append_results", fake_append_results)
    monkeypatch.setattr(
        "app.services.extraction_tier3.append_method_relations",
        fake_append_method_relations,
    )

    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1, 2],
        "sections": sections,
        "tables": [],
        "triple_candidates": triple_candidates,
    }

    summary = await run_tier3_verifier(paper_id, base_summary=base_summary)

    assert captured_relations, "Qualitative relation with strong signals should persist"
    relation = captured_relations[0]
    assert relation.dataset_id is not None
    assert relation.relation_type.value == "evaluates_on"
    assert relation.confidence == pytest.approx(0.51, rel=1e-3)
    assert relation.evidence and relation.evidence[0]["snippet"].startswith("QualNet is evaluated")

    meta = summary["metadata"]["tier3"]
    assert meta.get("persisted_relations") == len(captured_relations)
