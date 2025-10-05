from __future__ import annotations

import hashlib
import logging

import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from uuid import UUID, uuid4

from app.models.ontology import (
    ConceptResolutionType,
    Dataset,
    Method,
    Metric,
    Result,
    ResultCreate,
    Task,
)
from app.models.section import Section
from app.services.mentions import MentionObservation, replace_mentions_for_paper
from app.services.ontology_store import (
    ensure_dataset,
    ensure_method,
    ensure_metric,
    ensure_task,
    replace_claims,
    replace_results,
)
from app.services.papers import get_paper
from app.services.sections import list_sections
from app.services.storage import download_pdf_from_storage
from app.services.domain_inference import infer_domain_key
from app.utils.text_sanitize import sanitize_text

try:  # pragma: no cover - optional dependency
    import pdfplumber  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore[assignment]


SNIPPET_WINDOW = 80
DEFAULT_CONFIDENCE = 0.6
DEFAULT_RESULT_METRIC_TERMS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1 score",
    "f1-score",
    "auc",
    "roc auc",
    "map",
    "ndcg",
    "bleu",
    "rouge",
    "meteor",
    "chrF",
    "chrF++",
    "top-1",
    "top-5",
    "mae",
    "rmse",
    "psnr",
    "dice",
    "iou",
)

EVALUATE_PATTERN = re.compile(
    r"(?:evaluate(?:d)?|evaluating|tested|test(?:ing)?|trained|training|measured|measuring|validated|validating|benchmarked)"
    r"(?:\s+[A-Za-z0-9\-]+){0,6}\s+(?:on|against|using)\s+"
    r"(?P<dataset>[A-Za-z0-9\-\+\/ ]{2,}?)(?=(?:[,.;]"
    r"|\s+(?:and|with|for|using|achiev(?:es|ing)?|reports?|showing|compared|where|versus)\b|$))",
    re.IGNORECASE,
)
PROPOSE_PATTERN = re.compile(
    r"we\s+(?:propose|introduce|present|develop|design|build)\s+(?P<method>[A-Z0-9][A-Z0-9\-\+ ]{2,})",
    re.IGNORECASE,
)
STATE_OF_THE_ART_PATTERN = re.compile(
    r"state[- ]of[- ]the[- ]art|\bsota\b",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SPECIAL_TOKEN_PATTERN = re.compile(
    r"(<pad>|<s>|</s>|<bos>|</bos>|<eos>|</eos>|<unk>|</unk>|\[cls\]|\[sep\]|\[pad\])",
    re.IGNORECASE,
)
_APPENDIX_TITLE_PATTERN = re.compile(r"\b(?:appendix|supplement|supplementary)\b", re.IGNORECASE)
_ATTENTION_VIZ_PATTERN = re.compile(
    r"\battention\b.*\b(?:map|maps|weight|weights|head|heads|visualization|visualizations|heatmap|heatmaps)\b",
    re.IGNORECASE,
)
_INPUT_INPUT_PATTERN = re.compile(r"input-?input\s+layer", re.IGNORECASE)
_CAPTION_LINE_PATTERN = re.compile(r"^(?:fig(?:ure)?|tab(?:le)?|table)\s+[\w-]+(?:[:.].*)?", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
CITATION_RE = re.compile(r"\[[^\]]+\]")
DEFINITION_RE = re.compile(r"\b(?:is|are)\s+defined\s+as\b", re.IGNORECASE)

logger = logging.getLogger(__name__)

_POOL_NOT_INITIALIZED_TOKEN = "Database pool not initialized"


def _is_pool_not_initialized_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and _POOL_NOT_INITIALIZED_TOKEN in str(exc)


def _dedupe_aliases(aliases: Iterable[str] | None) -> list[str]:
    if not aliases:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for alias in aliases:
        text = (alias or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _fallback_method(name: str, *, aliases: Iterable[str] | None, description: Optional[str]) -> Method:
    now = datetime.now(timezone.utc)
    return Method(
        id=uuid4(),
        name=name,
        aliases=_dedupe_aliases(aliases),
        description=description,
        created_at=now,
        updated_at=now,
    )


def _fallback_dataset(
    name: str, *, aliases: Iterable[str] | None, description: Optional[str]
) -> Dataset:
    now = datetime.now(timezone.utc)
    return Dataset(
        id=uuid4(),
        name=name,
        aliases=_dedupe_aliases(aliases),
        description=description,
        created_at=now,
        updated_at=now,
    )


def _fallback_metric(
    name: str,
    *,
    unit: Optional[str],
    aliases: Iterable[str] | None,
    description: Optional[str],
) -> Metric:
    now = datetime.now(timezone.utc)
    return Metric(
        id=uuid4(),
        name=name,
        unit=unit,
        aliases=_dedupe_aliases(aliases),
        description=description,
        created_at=now,
        updated_at=now,
    )


def _fallback_task(name: str, *, aliases: Iterable[str] | None, description: Optional[str]) -> Task:
    now = datetime.now(timezone.utc)
    return Task(
        id=uuid4(),
        name=name,
        aliases=_dedupe_aliases(aliases),
        description=description,
        created_at=now,
        updated_at=now,
    )


def _fallback_result_from_create(result: ResultCreate) -> Result:
    now = datetime.now(timezone.utc)
    return Result(
        id=uuid4(),
        paper_id=result.paper_id,
        method_id=result.method_id,
        dataset_id=result.dataset_id,
        metric_id=result.metric_id,
        task_id=result.task_id,
        split=result.split,
        value_numeric=result.value_numeric,
        value_text=result.value_text,
        is_sota=result.is_sota,
        confidence=result.confidence,
        evidence=list(result.evidence),
        verified=result.verified,
        verifier_notes=result.verifier_notes,
        created_at=now,
        updated_at=now,
    )


async def _ensure_method_with_fallback(
    name: str,
    *,
    aliases: Iterable[str] | None = None,
    description: Optional[str] = None,
) -> Method:
    try:
        return await ensure_method(name, aliases=aliases, description=description)
    except RuntimeError as exc:
        if not _is_pool_not_initialized_error(exc):
            raise
        logger.debug("[tier1] using fallback method model for %s", name)
        return _fallback_method(name, aliases=aliases, description=description)


async def _ensure_dataset_with_fallback(
    name: str,
    *,
    aliases: Iterable[str] | None = None,
    description: Optional[str] = None,
) -> Dataset:
    try:
        return await ensure_dataset(name, aliases=aliases, description=description)
    except RuntimeError as exc:
        if not _is_pool_not_initialized_error(exc):
            raise
        logger.debug("[tier1] using fallback dataset model for %s", name)
        return _fallback_dataset(name, aliases=aliases, description=description)


async def _ensure_metric_with_fallback(
    name: str,
    *,
    unit: Optional[str] = None,
    aliases: Iterable[str] | None = None,
    description: Optional[str] = None,
) -> Metric:
    try:
        return await ensure_metric(
            name,
            unit=unit,
            aliases=aliases,
            description=description,
        )
    except RuntimeError as exc:
        if not _is_pool_not_initialized_error(exc):
            raise
        logger.debug("[tier1] using fallback metric model for %s", name)
        return _fallback_metric(
            name,
            unit=unit,
            aliases=aliases,
            description=description,
        )


async def _ensure_task_with_fallback(
    name: str,
    *,
    aliases: Iterable[str] | None = None,
    description: Optional[str] = None,
) -> Task:
    try:
        return await ensure_task(name, aliases=aliases, description=description)
    except RuntimeError as exc:
        if not _is_pool_not_initialized_error(exc):
            raise
        logger.debug("[tier1] using fallback task model for %s", name)
        return _fallback_task(name, aliases=aliases, description=description)



@dataclass
class LexiconEntry:
    name: str
    aliases: tuple[str, ...] = ()
    description: Optional[str] = None
    unit: Optional[str] = None
    patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)

    @property
    def phrases(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LexiconEntry:
        name = payload.get("name", "").strip()
        if not name:
            raise ValueError("Lexicon entry must include a name")
        aliases = tuple(
            alias.strip()
            for alias in payload.get("aliases", [])
            if isinstance(alias, str) and alias.strip()
        )
        description = payload.get("description")
        unit = payload.get("unit")
        patterns = _compile_patterns((name, *aliases))
        return cls(
            name=name,
            aliases=aliases,
            description=description,
            unit=unit,
            patterns=patterns,
        )


@dataclass
class DetectedEntity:
    name: str
    aliases: list[str] = field(default_factory=list)
    unit: Optional[str] = None
    description: Optional[str] = None
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def add_evidence(self, evidence: dict[str, Any]) -> None:
        self.evidence.append(evidence)


@dataclass
class DetectedResult:
    metric_name: str
    metric_aliases: list[str] = field(default_factory=list)
    metric_unit: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_aliases: list[str] = field(default_factory=list)
    method_name: Optional[str] = None
    method_aliases: list[str] = field(default_factory=list)
    task_name: Optional[str] = None
    value_text: str = ""
    value_numeric: Optional[float] = None
    split: Optional[str] = None
    is_sota: bool = False
    confidence: float = DEFAULT_CONFIDENCE
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Tier1Artifacts:
    methods: dict[str, DetectedEntity] = field(default_factory=dict)
    datasets: dict[str, DetectedEntity] = field(default_factory=dict)
    metrics: dict[str, DetectedEntity] = field(default_factory=dict)
    tasks: dict[str, DetectedEntity] = field(default_factory=dict)
    results: list[DetectedResult] = field(default_factory=list)

TableCell = dict[str, Any]
TableRecord = dict[str, Any]


class Tier1Lexicon:
    def __init__(
        self,
        *,
        methods: Sequence[LexiconEntry],
        datasets: Sequence[LexiconEntry],
        metrics: Sequence[LexiconEntry],
        tasks: Sequence[LexiconEntry],
        domain: Optional[str] = None,
    ) -> None:
        self.domain = domain
        self.methods = tuple(methods)
        self.datasets = tuple(datasets)
        self.metrics = tuple(metrics)
        self.tasks = tuple(tasks)

        self._method_lookup = _build_lookup(self.methods)
        self._dataset_lookup = _build_lookup(self.datasets)
        self._metric_lookup = _build_lookup(self.metrics)
        self._task_lookup = _build_lookup(self.tasks)
        self._result_pattern = _build_result_pattern(self.metrics)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Tier1Lexicon:
        methods = [LexiconEntry.from_payload(item) for item in payload.get("methods", [])]
        datasets = [LexiconEntry.from_payload(item) for item in payload.get("datasets", [])]
        metrics = [LexiconEntry.from_payload(item) for item in payload.get("metrics", [])]
        tasks = [LexiconEntry.from_payload(item) for item in payload.get("tasks", [])]
        return cls(
            methods=methods,
            datasets=datasets,
            metrics=metrics,
            tasks=tasks,
            domain=payload.get("domain"),
        )

    def lookup_method(self, phrase: str) ->Optional[LexiconEntry]:
        return self._method_lookup.get(_normalize_text(phrase))

    def lookup_dataset(self, phrase: str) ->Optional[LexiconEntry]:
        return self._dataset_lookup.get(_normalize_text(phrase))

    def lookup_metric(self, phrase: str) ->Optional[LexiconEntry]:
        return self._metric_lookup.get(_normalize_text(phrase))

    def lookup_task(self, phrase: str) ->Optional[LexiconEntry]:
        return self._task_lookup.get(_normalize_text(phrase))

    def find_method_in_text(self, text: str) ->Optional[LexiconEntry]:
        return _find_best_match(self._method_lookup, text)

    def find_dataset_in_text(self, text: str) ->Optional[LexiconEntry]:
        return _find_best_match(self._dataset_lookup, text)

    def find_metric_in_text(self, text: str) ->Optional[LexiconEntry]:
        return _find_best_match(self._metric_lookup, text)

    def find_task_in_text(self, text: str) ->Optional[LexiconEntry]:
        return _find_best_match(self._task_lookup, text)

    @property
    def result_pattern(self) -> re.Pattern[str]:
        return self._result_pattern


_LEXICON_CACHE: dict[str, Tier1Lexicon] = {}
_DOMAIN_ALIASES = {
    "mt": "machine_translation",
    "machine-translation": "machine_translation",
    "machine_translation": "machine_translation",
    "translation": "machine_translation",
    "general": "general_science",
    "general-science": "general_science",
}
_DOMAIN_FILE_OPTIONS = {
    "machine_translation": ["machine_translation_lexicon.json", "mt_lexicon.json"],
    "general_science": ["default_lexicon.json"],
    "biology": ["biology_lexicon.json"],
    "materials": ["materials_lexicon.json"],
}
_DEFAULT_LEXICON_FILES = ["default_lexicon.json", "mt_lexicon.json"]


def _normalize_domain_key(domain: Optional[str]) -> str:
    if not domain:
        return "default"
    normalized = re.sub(r"[^a-z0-9]+", "_", domain.strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        return "default"
    return _DOMAIN_ALIASES.get(normalized, normalized)


def load_lexicon(domain: Optional[str] = None) -> Tier1Lexicon:
    normalized = _normalize_domain_key(domain)
    cached = _LEXICON_CACHE.get(normalized)
    if cached is not None:
        return cached

    base_path = Path(__file__).resolve().parents[1] / "data"
    search_files: list[str] = []
    if normalized != "default":
        search_files.extend(_DOMAIN_FILE_OPTIONS.get(normalized, []))
    search_files.extend(_DEFAULT_LEXICON_FILES)

    for filename in search_files:
        path = base_path / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        lexicon = Tier1Lexicon.from_json(payload)
        _LEXICON_CACHE[normalized] = lexicon
        # Cache the default lexicon separately to avoid re-loading on future fallbacks.
        if normalized != "default" and "domain" in payload:
            fallback_key = _normalize_domain_key(payload.get("domain"))
            _LEXICON_CACHE.setdefault(fallback_key, lexicon)
        if normalized != "default" and filename in _DEFAULT_LEXICON_FILES:
            _LEXICON_CACHE.setdefault("default", lexicon)
        return lexicon

    # If no file exists for the requested domain, fall back to the default cache if available.
    default_lexicon = _LEXICON_CACHE.get("default")
    if default_lexicon is not None:
        _LEXICON_CACHE[normalized] = default_lexicon
        return default_lexicon

    fallback_name = _DEFAULT_LEXICON_FILES[0]
    default_path = base_path / fallback_name
    if not default_path.exists():
        raise FileNotFoundError("Tier-1 lexicon files are missing from the data directory")
    with default_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    lexicon = Tier1Lexicon.from_json(payload)
    _LEXICON_CACHE["default"] = lexicon
    _LEXICON_CACHE[normalized] = lexicon
    return lexicon


def load_mt_lexicon() -> Tier1Lexicon:
    return load_lexicon("machine_translation")


def _build_result_pattern(metrics: Sequence[LexiconEntry]) -> re.Pattern[str]:
    metric_terms: set[str] = set(DEFAULT_RESULT_METRIC_TERMS)
    for entry in metrics:
        for phrase in entry.phrases:
            cleaned = phrase.strip()
            if cleaned:
                metric_terms.add(cleaned)

    sorted_terms = sorted(metric_terms, key=lambda value: (-len(value), value.lower()))
    escaped_terms = [re.escape(term) for term in sorted_terms if term]
    if not escaped_terms:
        escaped_terms = [r"[A-Za-z][A-Za-z0-9_\-]{2,}"]
    metric_group = "|".join(escaped_terms)

    value_pattern = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    suffix_pattern = r"%|percent|pts?|pp|percentage points|score|db"
    pattern = re.compile(
        rf"(?P<metric>{metric_group})\s*(?:=|:|≈|~|was|is|of|score(?:s)?|achieve(?:s|d)?|reaches|yields|at)?\s*"
        rf"(?P<val>{value_pattern})\s*(?P<suffix>{suffix_pattern})?",
        re.IGNORECASE,
    )
    return pattern


def _clean_section_text(section: Section) -> str:
    content = section.content or ""
    if not content:
        return ""

    title_lower = (section.title or "").lower()
    cleaned_lines: list[str] = []
    for raw_line in content.splitlines():
        if not raw_line:
            continue
        if _SPECIAL_TOKEN_PATTERN.search(raw_line):
            continue
        normalized = sanitize_text(raw_line)
        if not normalized:
            continue
        if _is_noise_line(normalized, title_lower):
            continue
        cleaned_lines.append(normalized)
    return "\n".join(cleaned_lines)


def _is_noise_line(line: str, title_lower: str) -> bool:
    lower_line = line.lower().strip()
    if not lower_line:
        return True

    if _CAPTION_LINE_PATTERN.match(lower_line):
        return True

    if _INPUT_INPUT_PATTERN.search(lower_line):
        return True

    if _ATTENTION_VIZ_PATTERN.search(lower_line):
        if "attention" in title_lower or _APPENDIX_TITLE_PATTERN.search(title_lower):
            return True
        attention_terms = ("visualization", "heatmap", "matrix", "weights", "head")
        if any(term in lower_line for term in attention_terms):
            return True
        if lower_line.count("attention") > 1:
            return True

    if _APPENDIX_TITLE_PATTERN.search(title_lower) and _CAPTION_LINE_PATTERN.match(lower_line):
        return True

    letters = sum(1 for char in line if char.isalpha())
    digits = sum(1 for char in line if char.isdigit())
    if letters == 0 and digits > 0:
        return True

    return False

def extract_signals(
    sections: Sequence[Section],
    *,
    lexicon: Optional[Tier1Lexicon] = None,
    table_texts: Optional[Sequence[str | TableRecord]] = None,
) -> Tier1Artifacts:
    lexicon = lexicon or load_lexicon()
    artifacts = Tier1Artifacts()

    text_sources: list[tuple[Optional[Section], str, str]] = []
    for section in sections:
        cleaned_text = _clean_section_text(section)
        if cleaned_text:
            text_sources.append((section, cleaned_text, "section"))

    for text in table_texts or []:
        cleaned = text.strip()
        if cleaned:
            text_sources.append((None, cleaned, "table"))

    for section, text, source in text_sources:
        _collect_lexicon_mentions(artifacts.methods, lexicon.methods, section, text, source)
        _collect_lexicon_mentions(artifacts.datasets, lexicon.datasets, section, text, source)
        _collect_lexicon_mentions(artifacts.metrics, lexicon.metrics, section, text, source)
        _collect_lexicon_mentions(artifacts.tasks, lexicon.tasks, section, text, source)

        _collect_dataset_patterns(artifacts.datasets, lexicon, section, text, source)
        _collect_method_patterns(artifacts.methods, lexicon, section, text, source)

    for section, text, source in text_sources:
        artifacts.results.extend(
            _extract_results_from_text(
                text,
                lexicon=lexicon,
                section=section,
                source=source,
                artifacts=artifacts,
            )
        )

    return artifacts


async def run_tier1_extraction(
    paper_id: UUID,
    *,
    lexicon: Optional[Tier1Lexicon] = None,
    table_texts: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    paper = await get_paper(paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    if lexicon is None:
        domain_key = infer_domain_key(paper)
        lexicon = load_lexicon(domain_key)

    sections = await list_sections(paper_id=paper_id, limit=500, offset=0)

    table_records: list[TableRecord] = []
    table_text_payloads: list[str] = []

    if table_texts:
        for value in table_texts:
            if isinstance(value, str):
                table_text_payloads.append(value)
            elif isinstance(value, dict):
                table_records.append(value)

    pdf_bytes: bytes = b""
    if getattr(paper, "file_path", None):
        try:
            pdf_bytes = await download_pdf_from_storage(paper.file_path)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - storage failures should not break extraction
            pdf_bytes = b""

    if not table_records:
        table_records = _extract_tables_with_coordinates(pdf_bytes)

    if not table_text_payloads and table_records:
        table_text_payloads = [
            _table_cells_to_text(record.get("cells", []))
            for record in table_records
            if record.get("cells")
        ]

    if not table_text_payloads and pdf_bytes:
        table_text_payloads = extract_text_from_pdf_tables(pdf_bytes)

    artifacts = extract_signals(sections, lexicon=lexicon, table_texts=table_text_payloads)
    structural_summary = _build_structural_summary(sections, table_records)
    try:
        summary = await _persist_artifacts(
            paper_id,
            artifacts,
            paper_year=paper.year,
        )
    except RuntimeError as exc:
        logger.error("[tier1] failed to persist artifacts for paper %s: %s", paper_id, exc)
        summary = _build_empty_summary(paper_id)
        tier1_meta = summary.setdefault("metadata", {}).setdefault("tier1", {})
        tier1_meta["persistence_error"] = str(exc)

    summary["sections"] = structural_summary["sections"]
    summary["tables"] = structural_summary["tables"]
    summary["citations"] = structural_summary["citations"]
    summary["definition_sentences"] = structural_summary["definition_sentences"]

    metadata = summary.setdefault("metadata", {})
    for key, value in structural_summary.get("metadata", {}).items():
        if isinstance(metadata.get(key), dict) and isinstance(value, dict):
            metadata[key].update(value)
        else:
            metadata[key] = value

    return summary



def extract_text_from_pdf_tables(pdf_bytes: bytes) -> list[str]:
    if not pdf_bytes or pdfplumber is None:
        return []

    try:  # pragma: no cover - pdfplumber is optional
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as document:  # type: ignore[arg-type]
            texts: list[str] = []
            for page in document.pages:
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    continue
                for table in tables:
                    if not table:
                        continue
                    for row in table:
                        cleaned_cells: list[str] = []
                        for cell in row:
                            if not isinstance(cell, str):
                                continue
                            cleaned = sanitize_text(cell)
                            if cleaned:
                                cleaned_cells.append(cleaned)
                        if cleaned_cells:
                            texts.append(" ".join(cleaned_cells))
            return texts
    except Exception:
        return []


def _extract_tables_with_coordinates(pdf_bytes: bytes) -> list[TableRecord]:
    if not pdf_bytes or pdfplumber is None:
        return []

    try:  # pragma: no cover - pdfplumber is optional
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as document:  # type: ignore[arg-type]
            tables: list[TableRecord] = []
            for page_index, page in enumerate(document.pages, start=1):
                try:
                    candidates = page.extract_tables() or []
                except Exception:
                    continue
                for table_index, table in enumerate(candidates, start=1):
                    cells: list[TableCell] = []
                    for row_index, row in enumerate(table):
                        if not row:
                            continue
                        for column_index, cell_text in enumerate(row):
                            if not isinstance(cell_text, str):
                                continue
                            cleaned = sanitize_text(cell_text)
                            if not cleaned:
                                continue
                            cells.append(
                                {
                                    "row": row_index,
                                    "column": column_index,
                                    "text": cleaned,
                                }
                            )
                    if not cells:
                        continue
                    tables.append(
                        {
                            "table_id": f"table_{page_index}_{table_index}",
                            "page_number": page_index,
                            "section_id": None,
                            "cells": cells,
                        }
                    )
            return tables
    except Exception:
        return []


def _table_cells_to_text(cells: Sequence[TableCell]) -> str:
    parts: list[str] = []
    for cell in cells:
        text = cell.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return " ".join(parts)


def _build_empty_summary(paper_id: UUID) -> dict[str, Any]:
    return {
        "paper_id": str(paper_id),
        "tiers": [1],
        "methods": [],
        "datasets": [],
        "metrics": [],
        "tasks": [],
        "results": [],
        "claims": [],
        "metadata": {
            "tier1": {
                "result_count": 0,
                "method_count": 0,
                "dataset_count": 0,
                "metric_count": 0,
                "task_count": 0,
            }
        },
    }


async def _persist_artifacts(
    paper_id: UUID,
    artifacts: Tier1Artifacts,
    *,
    paper_year: Optional[int] = None,
) -> dict[str, Any]:
    method_cache: dict[str, Method] = {}
    dataset_cache: dict[str, Dataset] = {}
    metric_cache: dict[str, Metric] = {}
    task_cache: dict[str, Task] = {}
    mention_records: list[MentionObservation] = []

    for key, entity in artifacts.methods.items():
        method_model = await _ensure_method_with_fallback(
            entity.name,
            aliases=entity.aliases,
            description=entity.description,
        )
        method_cache[key] = method_model
        _collect_mentions_for_entity(
            mention_records,
            entity=entity,
            entity_id=method_model.id,
            resolution_type=ConceptResolutionType.METHOD,
            paper_id=paper_id,
            paper_year=paper_year,
        )

    for key, entity in artifacts.datasets.items():
        dataset_model = await _ensure_dataset_with_fallback(
            entity.name,
            aliases=entity.aliases,
            description=entity.description,
        )
        dataset_cache[key] = dataset_model
        _collect_mentions_for_entity(
            mention_records,
            entity=entity,
            entity_id=dataset_model.id,
            resolution_type=ConceptResolutionType.DATASET,
            paper_id=paper_id,
            paper_year=paper_year,
        )

    for key, entity in artifacts.metrics.items():
        metric_model = await _ensure_metric_with_fallback(
            entity.name,
            unit=entity.unit,
            aliases=entity.aliases,
            description=entity.description,
        )
        metric_cache[key] = metric_model
        _collect_mentions_for_entity(
            mention_records,
            entity=entity,
            entity_id=metric_model.id,
            resolution_type=ConceptResolutionType.METRIC,
            paper_id=paper_id,
            paper_year=paper_year,
        )

    for key, entity in artifacts.tasks.items():
        task_model = await _ensure_task_with_fallback(
            entity.name,
            aliases=entity.aliases,
            description=entity.description,
        )
        task_cache[key] = task_model
        _collect_mentions_for_entity(
            mention_records,
            entity=entity,
            entity_id=task_model.id,
            resolution_type=ConceptResolutionType.TASK,
            paper_id=paper_id,
            paper_year=paper_year,
        )

    result_models: list[ResultCreate] = []
    for detected in artifacts.results:
        method_id = None
        dataset_id = None
        metric_id = None
        task_id = None

        if detected.method_name:
            method_key = _normalize_text(detected.method_name)
            method_entity = artifacts.methods.get(method_key)
            if method_entity is None:
                method_entity = DetectedEntity(
                    detected.method_name,
                    aliases=list(dict.fromkeys(detected.method_aliases)),
                )
                artifacts.methods[method_key] = method_entity
            method_model = method_cache.get(method_key)
            if method_model is None:
                method_model = await _ensure_method_with_fallback(
                    method_entity.name,
                    aliases=method_entity.aliases,
                    description=method_entity.description,
                )
                method_cache[method_key] = method_model
                _collect_mentions_for_entity(
                    mention_records,
                    entity=method_entity,
                    entity_id=method_model.id,
                    resolution_type=ConceptResolutionType.METHOD,
                    paper_id=paper_id,
                    paper_year=paper_year,
                )
            method_id = method_model.id

        if detected.dataset_name:
            dataset_key = _normalize_text(detected.dataset_name)
            dataset_entity = artifacts.datasets.get(dataset_key)
            if dataset_entity is None:
                dataset_entity = DetectedEntity(
                    detected.dataset_name,
                    aliases=list(dict.fromkeys(detected.dataset_aliases)),
                )
                artifacts.datasets[dataset_key] = dataset_entity
            dataset_model = dataset_cache.get(dataset_key)
            if dataset_model is None:
                dataset_model = await _ensure_dataset_with_fallback(
                    dataset_entity.name,
                    aliases=dataset_entity.aliases,
                    description=dataset_entity.description,
                )
                dataset_cache[dataset_key] = dataset_model
                _collect_mentions_for_entity(
                    mention_records,
                    entity=dataset_entity,
                    entity_id=dataset_model.id,
                    resolution_type=ConceptResolutionType.DATASET,
                    paper_id=paper_id,
                    paper_year=paper_year,
                )
            dataset_id = dataset_model.id

        if detected.metric_name:
            metric_key = _normalize_text(detected.metric_name)
            metric_entity = artifacts.metrics.get(metric_key)
            if metric_entity is None:
                metric_entity = DetectedEntity(
                    detected.metric_name,
                    aliases=list(dict.fromkeys(detected.metric_aliases)),
                    unit=detected.metric_unit,
                )
                artifacts.metrics[metric_key] = metric_entity
            metric_model = metric_cache.get(metric_key)
            if metric_model is None:
                metric_model = await _ensure_metric_with_fallback(
                    metric_entity.name,
                    unit=metric_entity.unit,
                    aliases=metric_entity.aliases,
                    description=metric_entity.description,
                )
                metric_cache[metric_key] = metric_model
                _collect_mentions_for_entity(
                    mention_records,
                    entity=metric_entity,
                    entity_id=metric_model.id,
                    resolution_type=ConceptResolutionType.METRIC,
                    paper_id=paper_id,
                    paper_year=paper_year,
                )
            metric_id = metric_model.id

        task_name = detected.task_name
        if not task_name and artifacts.tasks:
            # Default to first detected task if none was explicitly linked.
            task_name = next(iter(artifacts.tasks.values())).name

        if task_name:
            task_key = _normalize_text(task_name)
            task_entity = artifacts.tasks.get(task_key)
            if task_entity is None:
                task_entity = DetectedEntity(task_name)
                artifacts.tasks[task_key] = task_entity
            task_model = task_cache.get(task_key)
            if task_model is None:
                task_model = await _ensure_task_with_fallback(
                    task_entity.name,
                    aliases=task_entity.aliases,
                    description=task_entity.description,
                )
                task_cache[task_key] = task_model
                _collect_mentions_for_entity(
                    mention_records,
                    entity=task_entity,
                    entity_id=task_model.id,
                    resolution_type=ConceptResolutionType.TASK,
                    paper_id=paper_id,
                    paper_year=paper_year,
                )
            task_id = task_model.id

        value_numeric = (
            Decimal(str(detected.value_numeric)) if detected.value_numeric is not None else None
        )

        result_models.append(
            ResultCreate(
                paper_id=paper_id,
                method_id=method_id,
                dataset_id=dataset_id,
                metric_id=metric_id,
                task_id=task_id,
                split=detected.split,
                value_numeric=value_numeric,
                value_text=detected.value_text or None,
                is_sota=detected.is_sota,
                confidence=detected.confidence,
                evidence=detected.evidence,
            )
        )

    try:
        stored_results = await replace_results(paper_id, result_models)
    except RuntimeError as exc:
        if not _is_pool_not_initialized_error(exc):
            raise
        logger.debug(
            "[tier1] using fallback result persistence for paper %s", paper_id
        )
        stored_results = [_fallback_result_from_create(result) for result in result_models]

    try:
        stored_claims = await replace_claims(paper_id, [])
    except RuntimeError as exc:
        if not _is_pool_not_initialized_error(exc):
            raise
        stored_claims = []

    method_by_id = {model.id: model for model in method_cache.values()}
    dataset_by_id = {model.id: model for model in dataset_cache.values()}
    metric_by_id = {model.id: model for model in metric_cache.values()}
    task_by_id = {model.id: model for model in task_cache.values()}

    try:
        await replace_mentions_for_paper(paper_id, mention_records)
    except RuntimeError as exc:
        if not _is_pool_not_initialized_error(exc):
            raise
        logger.debug(
            "[tier1] skipping mention persistence for paper %s due to missing pool",
            paper_id,
        )

    return {
        "paper_id": str(paper_id),
        "tiers": [1],
        "methods": [_serialize_method(model) for model in method_cache.values()],
        "datasets": [_serialize_dataset(model) for model in dataset_cache.values()],
        "metrics": [_serialize_metric(model) for model in metric_cache.values()],
        "tasks": [_serialize_task(model) for model in task_cache.values()],
        "results": [
            _serialize_result(
                result,
                method_by_id=method_by_id,
                dataset_by_id=dataset_by_id,
                metric_by_id=metric_by_id,
                task_by_id=task_by_id,
            )
            for result in stored_results
        ],
        "claims": [_serialize_claim(claim) for claim in stored_claims],
        "metadata": {
            "tier1": {
                "result_count": len(stored_results),
                "method_count": len(method_cache),
                "dataset_count": len(dataset_cache),
                "metric_count": len(metric_cache),
                "task_count": len(task_cache),
            }
        },
    }


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_uuid(value: Any) -> Optional[UUID]:
    if not value:
        return None
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


def _collect_mentions_for_entity(
    mentions: list[MentionObservation],
    *,
    entity: DetectedEntity,
    entity_id: UUID,
    resolution_type: ConceptResolutionType,
    paper_id: UUID,
    paper_year: Optional[int],
) -> None:
    if not entity.evidence:
        mentions.append(
            MentionObservation(
                resolution_type=resolution_type,
                entity_id=entity_id,
                paper_id=paper_id,
                section_id=None,
                surface=entity.name,
                mention_type=resolution_type.value,
                snippet=None,
                start=None,
                end=None,
                source=None,
                first_seen_year=paper_year,
            )
        )
        return

    for evidence in entity.evidence:
        surface = evidence.get("mention_text") if isinstance(evidence, dict) else None
        if not isinstance(surface, str) or not surface.strip():
            surface = entity.name
        char_range = evidence.get("range") if isinstance(evidence, dict) else None
        start: Optional[int] = None
        end: Optional[int] = None
        if isinstance(char_range, (list, tuple)):
            if char_range:
                start = _safe_int(char_range[0])
            if len(char_range) >= 2:
                end = _safe_int(char_range[1])
        snippet = evidence.get("snippet") if isinstance(evidence, dict) else None
        source = evidence.get("source") if isinstance(evidence, dict) else None
        mentions.append(
            MentionObservation(
                resolution_type=resolution_type,
                entity_id=entity_id,
                paper_id=paper_id,
                section_id=_parse_optional_uuid(
                    evidence.get("section_id") if isinstance(evidence, dict) else None
                ),
                surface=surface,
                mention_type=resolution_type.value,
                snippet=snippet if isinstance(snippet, str) else None,
                start=start,
                end=end,
                source=source if isinstance(source, str) else None,
                first_seen_year=paper_year,
            )
        )


def _build_structural_summary(
    sections: Sequence[Section],
    tables: Sequence[TableRecord],
) -> dict[str, Any]:
    sections_payload: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    definition_sentences: list[dict[str, Any]] = []

    tables_payload = []
    tables_by_section: dict[str, list[str]] = {}
    for table in tables:
        normalized = {
            "table_id": table.get("table_id"),
            "page_number": table.get("page_number"),
            "section_id": table.get("section_id"),
            "cells": list(table.get("cells", [])),
        }
        tables_payload.append(normalized)
        section_id = normalized.get("section_id")
        table_id = normalized.get("table_id")
        if section_id and table_id:
            section_key = str(section_id)
            tables_by_section.setdefault(section_key, []).append(str(table_id))

    for section in sections:
        payload, section_citations, section_definitions = _build_section_payload(
            section,
            table_refs=tables_by_section.get(str(section.id), []),
        )
        if not payload:
            continue
        sections_payload.append(payload)
        citations.extend(section_citations)
        definition_sentences.extend(section_definitions)

    return {
        "sections": sections_payload,
        "tables": tables_payload,
        "citations": citations,
        "definition_sentences": definition_sentences,
        "metadata": {"section_count": len(sections_payload)},
    }


def _build_section_payload(
    section: Section,
    *,
    table_refs: Optional[Sequence[str]] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cleaned_text = _clean_section_text(section)
    if not cleaned_text:
        return None, [], []

    normalized_text = re.sub(r"\s+", " ", cleaned_text).strip()
    if not normalized_text:
        return None, [], []

    section_hash = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
    sentences: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    definition_sentences: list[dict[str, Any]] = []

    for sentence_index, match in enumerate(SENTENCE_RE.finditer(normalized_text)):
        raw_sentence = match.group().strip()
        if not raw_sentence:
            continue
        start = match.start()
        end = match.end()
        sentence_entry = {
            "sentence_index": sentence_index,
            "start": start,
            "end": end,
            "text": raw_sentence,
        }

        has_citation = False
        for citation in CITATION_RE.finditer(raw_sentence):
            has_citation = True
            citations.append(
                {
                    "section_id": str(section.id),
                    "sentence_index": sentence_index,
                    "text": citation.group(),
                    "start": start + citation.start(),
                    "end": start + citation.end(),
                }
            )
        if has_citation:
            sentence_entry["contains_citation"] = True

        if DEFINITION_RE.search(raw_sentence):
            sentence_entry["is_definition"] = True
            definition_sentences.append(
                {
                    "section_id": str(section.id),
                    "sentence_index": sentence_index,
                    "text": raw_sentence,
                }
            )

        sentences.append(sentence_entry)

    if not sentences:
        return None, [], []

    sentence_spans: list[dict[str, Any]] = []
    for entry in sentences:
        span = {
            "sentence_index": entry["sentence_index"],
            "start": entry["start"],
            "end": entry["end"],
            "text": entry["text"],
        }
        if entry.get("is_definition"):
            span["is_definition"] = True
        if entry.get("contains_citation"):
            span["contains_citation"] = True
        sentence_spans.append(span)

    payload = {
        "section_id": str(section.id),
        "section_hash": section_hash,
        "title": section.title,
        "page_number": section.page_number,
        "char_start": section.char_start,
        "char_end": section.char_end,
        "sentence_spans": sentence_spans,
        "captions": [],
        "table_refs": [str(ref) for ref in (table_refs or [])],
    }

    return payload, citations, definition_sentences


def _collect_lexicon_mentions(
    target: dict[str, DetectedEntity],
    entries: Sequence[LexiconEntry],
    section: Optional[Section],
    text: str,
    source: str,
) -> None:
    for entry in entries:
        for pattern in entry.patterns:
            for match in pattern.finditer(text):
                _record_entity(
                    target,
                    name=entry.name,
                    aliases=entry.aliases,
                    unit=entry.unit,
                    description=entry.description,
                    section=section,
                    text=text,
                    start=match.start(),
                    end=match.end(),
                    source=source,
                )


def _collect_dataset_patterns(
    target: dict[str, DetectedEntity],
    lexicon: Tier1Lexicon,
    section: Optional[Section],
    text: str,
    source: str,
) -> None:
    for match in EVALUATE_PATTERN.finditer(text):
        raw = match.group("dataset")
        cleaned = _clean_dataset_name(raw)
        if not cleaned:
            continue
        entry = lexicon.lookup_dataset(cleaned) or lexicon.find_dataset_in_text(cleaned)
        aliases: Iterable[str] = entry.aliases if entry else []
        description = entry.description if entry else None
        name = entry.name if entry else cleaned
        _record_entity(
            target,
            name=name,
            aliases=aliases,
            unit=None,
            description=description,
            section=section,
            text=text,
            start=match.start("dataset"),
            end=match.end("dataset"),
            source=source,
        )


def _collect_method_patterns(
    target: dict[str, DetectedEntity],
    lexicon: Tier1Lexicon,
    section: Optional[Section],
    text: str,
    source: str,
) -> None:
    for match in PROPOSE_PATTERN.finditer(text):
        raw = match.group("method")
        cleaned = raw.strip(" .,;:")
        if not cleaned:
            continue
        entry = lexicon.lookup_method(cleaned) or lexicon.find_method_in_text(cleaned)
        aliases: Iterable[str] = entry.aliases if entry else []
        description = entry.description if entry else None
        name = entry.name if entry else cleaned
        _record_entity(
            target,
            name=name,
            aliases=aliases,
            unit=None,
            description=description,
            section=section,
            text=text,
            start=match.start("method"),
            end=match.end("method"),
            source=source,
        )


def _extract_results_from_text(
    text: str,
    *,
    lexicon: Tier1Lexicon,
    section: Optional[Section],
    source: str,
    artifacts: Tier1Artifacts,
) -> list[DetectedResult]:
    results: list[DetectedResult] = []
    context_start = 0
    context_end = len(text)
    for match in lexicon.result_pattern.finditer(text):
        metric_phrase = match.group("metric")
        metric_entry = lexicon.lookup_metric(metric_phrase) or lexicon.find_metric_in_text(metric_phrase)
        if metric_entry:
            metric_name = metric_entry.name
            metric_aliases = list(metric_entry.aliases)
            metric_unit = metric_entry.unit
        else:
            metric_name = metric_phrase.strip()
            metric_aliases = []
            metric_unit = None

        _record_entity(
            artifacts.metrics,
            name=metric_name,
            aliases=metric_aliases,
            unit=metric_unit,
            description=metric_entry.description if metric_entry else None,
            section=section,
            text=text,
            start=match.start("metric"),
            end=match.end("metric"),
            source=source,
        )

        value_raw = match.group("val")
        suffix = (match.group("suffix") or "").strip()
        value_text = f"{value_raw}{suffix}".strip()
        try:
            value_numeric = float(value_raw)
        except ValueError:
            value_numeric = None

        window_start = max(context_start, match.start() - SNIPPET_WINDOW)
        window_end = min(context_end, match.end() + SNIPPET_WINDOW)
        snippet = text[window_start:window_end]
        evidence = [
            _build_evidence(
                section=section,
                text=text,
                start=match.start(),
                end=match.end(),
                source=source,
            )
        ]

        dataset_entry = lexicon.find_dataset_in_text(snippet)
        dataset_name = None
        dataset_aliases: list[str] = []
        if dataset_entry:
            dataset_name = dataset_entry.name
            dataset_aliases = list(dataset_entry.aliases)
            dataset_span = _locate_pattern(dataset_entry.patterns, text, window_start, window_end)
            if dataset_span:
                _record_entity(
                    artifacts.datasets,
                    name=dataset_entry.name,
                    aliases=dataset_entry.aliases,
                    unit=None,
                    description=dataset_entry.description,
                    section=section,
                    text=text,
                    start=dataset_span[0],
                    end=dataset_span[1],
                    source=source,
                )
        elif source == "table":
            dataset_entity = _find_entity_with_source(artifacts.datasets, source)
            if dataset_entity:
                dataset_name = dataset_entity.name
                dataset_aliases = dataset_entity.aliases
        elif not dataset_name:
            table_dataset = _find_entity_with_source(artifacts.datasets, "table")
            if table_dataset:
                dataset_name = table_dataset.name
                dataset_aliases = table_dataset.aliases
        elif len(artifacts.datasets) == 1:
            only_dataset = next(iter(artifacts.datasets.values()))
            dataset_name = only_dataset.name
            dataset_aliases = only_dataset.aliases
        if dataset_name is None:
            nearby_dataset = _find_entity_near_span(
                artifacts.datasets,
                section,
                match.start(),
            )
            if nearby_dataset:
                dataset_name = nearby_dataset.name
                dataset_aliases = nearby_dataset.aliases

        method_entry = lexicon.find_method_in_text(snippet)
        method_name = None
        method_aliases: list[str] = []
        if method_entry:
            method_name = method_entry.name
            method_aliases = list(method_entry.aliases)
            method_span = _locate_pattern(method_entry.patterns, text, window_start, window_end)
            if method_span:
                _record_entity(
                    artifacts.methods,
                    name=method_entry.name,
                    aliases=method_entry.aliases,
                    unit=None,
                    description=method_entry.description,
                    section=section,
                    text=text,
                    start=method_span[0],
                    end=method_span[1],
                    source=source,
                )
        elif source == "table":
            method_entity = _find_entity_with_source(artifacts.methods, source)
            if method_entity:
                method_name = method_entity.name
                method_aliases = method_entity.aliases
        elif len(artifacts.methods) == 1:
            only_method = next(iter(artifacts.methods.values()))
            method_name = only_method.name
            method_aliases = only_method.aliases
        if method_name is None:
            nearby_method = _find_entity_near_span(
                artifacts.methods,
                section,
                match.start(),
            )
            if nearby_method:
                method_name = nearby_method.name
                method_aliases = nearby_method.aliases

        task_entry = lexicon.find_task_in_text(snippet)
        task_name = task_entry.name if task_entry else None
        if task_entry:
            task_span = _locate_pattern(task_entry.patterns, text, window_start, window_end)
            if task_span:
                _record_entity(
                    artifacts.tasks,
                    name=task_entry.name,
                    aliases=task_entry.aliases,
                    unit=None,
                    description=task_entry.description,
                    section=section,
                    text=text,
                    start=task_span[0],
                    end=task_span[1],
                    source=source,
                )
        if task_name is None:
            nearby_task = _find_entity_near_span(
                artifacts.tasks,
                section,
                match.start(),
            )
            if nearby_task:
                task_name = nearby_task.name

        split = _infer_split(snippet)
        is_sota = bool(STATE_OF_THE_ART_PATTERN.search(snippet))

        results.append(
            DetectedResult(
                metric_name=metric_name,
                metric_aliases=metric_aliases,
                metric_unit=metric_unit,
                dataset_name=dataset_name,
                dataset_aliases=dataset_aliases,
                method_name=method_name,
                method_aliases=method_aliases,
                task_name=task_name,
                value_text=value_text,
                value_numeric=value_numeric,
                split=split,
                is_sota=is_sota,
                evidence=evidence,
            )
        )

    return results


def _record_entity(
    target: dict[str, DetectedEntity],
    *,
    name: str,
    aliases: Iterable[str],
    unit: Optional[str],
    description: Optional[str],
    section: Optional[Section],
    text: str,
    start: int,
    end: int,
    source: str,
) -> None:
    key = _normalize_text(name)
    if not key:
        return
    alias_list = list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))
    entity = target.get(key)
    if entity is None:
        entity = DetectedEntity(
            name=name,
            aliases=alias_list,
            unit=unit,
            description=description,
        )
        target[key] = entity
    elif alias_list:
        merged = list(dict.fromkeys([*entity.aliases, *alias_list]))
        entity.aliases = merged
    evidence = _build_evidence(section=section, text=text, start=start, end=end, source=source)
    entity.add_evidence(evidence)


def _build_evidence(
    *,
    section: Optional[Section],
    text: str,
    start: int,
    end: int,
    source: str,
) -> dict[str, Any]:
    snippet_start = max(0, start - SNIPPET_WINDOW)
    snippet_end = min(len(text), end + SNIPPET_WINDOW)
    snippet = text[snippet_start:snippet_end].strip()
    safe_start = max(0, min(len(text), start))
    safe_end = max(safe_start, min(len(text), end))
    mention_text = text[safe_start:safe_end].strip()
    return {
        "section_id": str(section.id) if section else None,
        "range": [start, end],
        "snippet": snippet,
        "page": section.page_number if section else None,
        "section_title": section.title if section else None,
        "source": source,
        "mention_text": mention_text,
    }


def _clean_dataset_name(name: str) -> str:
    cleaned = name.strip(" .,;:\n\t")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(dataset|corpus|benchmark|task)\b", "", cleaned, flags=re.IGNORECASE)
    parts = re.split(
        r"\b(?:and|with|for|using|achiev(?:es|ing)?|reports?|showing|compared|where)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    cleaned = parts[0]
    cleaned = re.sub(r"[(),]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        lowered = cleaned.lower()
        if not any(ch.isdigit() for ch in cleaned):
            disqualifiers = {
                "model",
                "method",
                "approach",
                "architecture",
                "network",
                "framework",
                "system",
                "technique",
                "baseline",
                "algorithm",
            }
            tokens = set(lowered.split())
            if tokens.intersection(disqualifiers):
                return ""
    return cleaned


def _normalize_text(value: str) -> str:
    normalized = _NON_ALNUM_RE.sub(" ", value.lower())
    return " ".join(normalized.split())


def _compile_patterns(phrases: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    patterns: list[re.Pattern[str]] = []
    for phrase in dict.fromkeys(phrases):
        cleaned = phrase.strip()
        if not cleaned:
            continue
        escaped = re.escape(cleaned)
        pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
        patterns.append(pattern)
    return tuple(patterns)


def _build_lookup(entries: Sequence[LexiconEntry]) -> dict[str, LexiconEntry]:
    lookup: dict[str, LexiconEntry] = {}
    for entry in entries:
        for phrase in entry.phrases:
            key = _normalize_text(phrase)
            if key:
                lookup[key] = entry
    return lookup


def _find_best_match(lookup: dict[str, LexiconEntry], text: str) ->Optional[LexiconEntry]:
    normalized = _normalize_text(text)
    best: Optional[LexiconEntry] = None
    best_length = 0
    for key, entry in lookup.items():
        if len(key) < 3:
            continue
        if key in normalized and len(key) > best_length:
            best = entry
            best_length = len(key)
    return best


def _infer_split(snippet: str) ->Optional[str]:
    lowered = snippet.lower()
    if "test" in lowered:
        return "test"
    if "dev" in lowered or "validation" in lowered:
        return "dev"
    if "train" in lowered:
        return "train"
    return None


def _locate_pattern(
    patterns: Sequence[re.Pattern[str]],
    text: str,
    start: int,
    end: int,
) ->Optional[tuple[int, int]]:
    for pattern in patterns:
        match = pattern.search(text, start, end)
        if match:
            return match.span()
    return None


def _find_entity_with_source(
    entities: dict[str, DetectedEntity],
    source: str,
) ->Optional[DetectedEntity]:
    for entity in entities.values():
        for ev in entity.evidence:
            if ev.get("source") == source:
                return entity
    return None


def _find_entity_near_span(
    entities: dict[str, DetectedEntity],
    section: Optional[Section],
    position: int,
    *,
    window: int = 160,
) ->Optional[DetectedEntity]:
    if section is None:
        return None
    section_id = str(section.id)
    for entity in entities.values():
        for evidence in entity.evidence:
            if evidence.get("section_id") != section_id:
                continue
            char_range = evidence.get("range")
            if (
                isinstance(char_range, list)
                and len(char_range) == 2
                and all(isinstance(value, int) for value in char_range)
            ):
                start, end = char_range
                if abs(start - position) <= window or abs(end - position) <= window:
                    return entity
    return None


def _serialize_method(method: Method) -> dict[str, Any]:
    return {
        "id": str(method.id),
        "name": method.name,
        "aliases": list(method.aliases),
        "description": method.description,
        "created_at": method.created_at.isoformat(),
        "updated_at": method.updated_at.isoformat(),
    }


def _serialize_dataset(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": str(dataset.id),
        "name": dataset.name,
        "aliases": list(dataset.aliases),
        "description": dataset.description,
        "created_at": dataset.created_at.isoformat(),
        "updated_at": dataset.updated_at.isoformat(),
    }


def _serialize_metric(metric: Metric) -> dict[str, Any]:
    return {
        "id": str(metric.id),
        "name": metric.name,
        "unit": metric.unit,
        "aliases": list(metric.aliases),
        "description": metric.description,
        "created_at": metric.created_at.isoformat(),
        "updated_at": metric.updated_at.isoformat(),
    }


def _serialize_task(task: Task) -> dict[str, Any]:
    return {
        "id": str(task.id),
        "name": task.name,
        "aliases": list(task.aliases),
        "description": task.description,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }


def _serialize_result(
    result: Result,
    *,
    method_by_id: dict[UUID, Method],
    dataset_by_id: dict[UUID, Dataset],
    metric_by_id: dict[UUID, Metric],
    task_by_id: dict[UUID, Task],
) -> dict[str, Any]:
    return {
        "id": str(result.id),
        "paper_id": str(result.paper_id),
        "method": _serialize_method(method_by_id[result.method_id]) if result.method_id else None,
        "dataset": _serialize_dataset(dataset_by_id[result.dataset_id]) if result.dataset_id else None,
        "metric": _serialize_metric(metric_by_id[result.metric_id]) if result.metric_id else None,
        "task": _serialize_task(task_by_id[result.task_id]) if result.task_id else None,
        "split": result.split,
        "value_numeric": float(result.value_numeric) if result.value_numeric is not None else None,
        "value_text": result.value_text,
        "is_sota": result.is_sota,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "verified": result.verified,
        "verifier_notes": result.verifier_notes,
        "created_at": result.created_at.isoformat(),
        "updated_at": result.updated_at.isoformat(),
    }


def _serialize_claim(claim: Claim) -> dict[str, Any]:
    return {
        "id": str(claim.id),
        "paper_id": str(claim.paper_id),
        "category": claim.category.value,
        "text": claim.text,
        "confidence": claim.confidence,
        "evidence": claim.evidence,
        "created_at": claim.created_at.isoformat(),
        "updated_at": claim.updated_at.isoformat(),
    }
