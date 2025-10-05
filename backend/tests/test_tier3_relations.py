from uuid import uuid4

import pytest

from app.services.extraction_tier3_relations import run_tier3_relations


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_run_tier3_relations_appends_fallback_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    base_summary = {
        "triple_candidates": [
            {
                "subject": "Baseline",
                "relation": "evaluated on",
                "object": "Dataset-X",
                "evidence": "Baseline evaluated on Dataset-X with accuracy.",
            }
        ],
        "metadata": {},
        "tiers": [1, 2],
    }

    fallback_candidate = {
        "subject": "AlphaNet",
        "relation": "evaluated on",
        "object": "CIFAR-10",
        "evidence": "AlphaNet was evaluated on the CIFAR-10 benchmark dataset.",
        "evidence_spans": [],
    }

    fallback_meta = {
        "status": "succeeded",
        "triggered": True,
        "accepted": 1,
        "errors": [],
    }

    async def fake_fallback(*args, **kwargs):  # type: ignore[override]
        return [fallback_candidate], fallback_meta

    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.maybe_apply_relation_llm_fallback",
        fake_fallback,
    )

    result = await run_tier3_relations(paper_id=uuid4(), base_summary=base_summary)

    assert result is not base_summary
    assert len(result["triple_candidates"]) == 2
    assert fallback_candidate in result["triple_candidates"]

    meta = result["metadata"].get("tier3_relations")
    assert meta["added_candidates"] == 1
    assert meta["fallback"] == fallback_meta
    assert meta["status"] == "succeeded"
    assert result["tiers"] == [1, 2, 3]


@pytest.mark.anyio
async def test_run_tier3_relations_records_disabled_status(monkeypatch: pytest.MonkeyPatch) -> None:
    base_summary = {
        "triple_candidates": [],
    }

    fallback_meta = {
        "status": "disabled",
        "enabled": False,
        "triggered": False,
        "accepted": 0,
        "errors": [],
    }

    async def fake_fallback(*args, **kwargs):  # type: ignore[override]
        return [], fallback_meta

    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.maybe_apply_relation_llm_fallback",
        fake_fallback,
    )

    result = await run_tier3_relations(paper_id=uuid4(), base_summary=base_summary)

    assert result["triple_candidates"] == []
    meta = result["metadata"]["tier3_relations"]
    assert meta["fallback"] == fallback_meta
    assert meta["status"] == "disabled"
    assert meta["added_candidates"] == 0


@pytest.mark.anyio
async def test_run_tier3_relations_handles_fallback_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    base_summary = {
        "triple_candidates": [],
    }

    async def fake_fallback(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.maybe_apply_relation_llm_fallback",
        fake_fallback,
    )

    result = await run_tier3_relations(paper_id=uuid4(), base_summary=base_summary)

    fallback_meta = result["metadata"]["tier3_relations"]["fallback"]
    assert fallback_meta["status"] == "error"
    assert fallback_meta["errors"] == ["boom"]
    assert result["metadata"]["tier3_relations"]["status"] == "error"


@pytest.mark.anyio
async def test_run_tier3_relations_requires_base_summary() -> None:
    with pytest.raises(ValueError):
        await run_tier3_relations(paper_id=uuid4(), base_summary=None)

