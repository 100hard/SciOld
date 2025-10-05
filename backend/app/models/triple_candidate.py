from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import UUID


@dataclass
class TripleCandidateRecord:
    paper_id: UUID
    section_id: Optional[str]
    subject: str
    relation: str
    object: str
    subject_span: list[int]
    object_span: list[int]
    subject_type_guess: str
    relation_type_guess: str
    object_type_guess: str
    evidence: str
    triple_conf: float
    schema_match_score: float
    tier: str
    graph_metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    confidence_components: Dict[str, Any] = field(default_factory=dict)
    verifier_notes: Optional[str] = None
