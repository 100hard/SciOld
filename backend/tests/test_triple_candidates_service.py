from __future__ import annotations

import asyncio
from typing import Any, Iterable
from uuid import UUID, uuid4

import pytest

from app.models.triple_candidate import TripleCandidateRecord
from app.services import triple_candidates


class _FakeTransaction:
    async def __aenter__(self) -> "_FakeTransaction":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeAcquire:
    def __init__(self, connection: "_FakeConnection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "_FakeConnection":
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeConnection:
    def __init__(self, *, section_ids: Iterable[UUID]) -> None:
        self.section_ids = list(section_ids)
        self.deleted: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_queries: list[tuple[str, tuple[Any, ...]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[Any, ...]]]] = []

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, sql: str, *args: Any) -> None:
        self.deleted.append((sql, args))

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, UUID]]:
        self.fetch_queries.append((sql, args))
        return [{"id": section_id} for section_id in self.section_ids]

    async def executemany(self, sql: str, records: Iterable[tuple[Any, ...]]) -> None:
        self.executemany_calls.append((sql, list(records)))


class _FakePool:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._connection)


def _make_candidate(paper_id: UUID, *, section_id: str | None) -> TripleCandidateRecord:
    return TripleCandidateRecord(
        paper_id=paper_id,
        section_id=section_id,
        subject="subject",
        relation="relation",
        object="object",
        subject_span=[0, 1],
        object_span=[2, 3],
        subject_type_guess="Concept",
        relation_type_guess="RelatedTo",
        object_type_guess="Concept",
        evidence="Example evidence",
        triple_conf=0.9,
        schema_match_score=0.75,
        tier="tier2",
        graph_metadata={},
        provenance={},
        verification={},
        confidence_components={},
        verifier_notes=None,
    )


def test_replace_triple_candidates_drops_unknown_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    valid_section = uuid4()
    invalid_section = uuid4()

    connection = _FakeConnection(section_ids=[valid_section])
    pool = _FakePool(connection)
    monkeypatch.setattr(triple_candidates, "get_pool", lambda: pool)

    candidates = [
        _make_candidate(paper_id, section_id=str(valid_section)),
        _make_candidate(paper_id, section_id=str(invalid_section)),
        _make_candidate(paper_id, section_id=None),
    ]

    asyncio.run(triple_candidates.replace_triple_candidates(paper_id, candidates))

    assert connection.fetch_queries == [
        ("SELECT id FROM sections WHERE paper_id = $1", (paper_id,))
    ]
    assert len(connection.executemany_calls) == 1
    _, records = connection.executemany_calls[0]
    assert [record[1] for record in records] == [valid_section, None, None]


def test_replace_triple_candidates_skips_lookup_without_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    connection = _FakeConnection(section_ids=[])
    pool = _FakePool(connection)
    monkeypatch.setattr(triple_candidates, "get_pool", lambda: pool)

    candidates = [
        _make_candidate(paper_id, section_id=None),
        _make_candidate(paper_id, section_id=""),
    ]

    asyncio.run(triple_candidates.replace_triple_candidates(paper_id, candidates))

    assert connection.fetch_queries == []
    assert len(connection.executemany_calls) == 1
    _, records = connection.executemany_calls[0]
    assert [record[1] for record in records] == [None, None]
