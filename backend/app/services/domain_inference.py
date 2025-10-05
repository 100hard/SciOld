"""Shared helpers for inferring coarse paper domains.

These utilities are reused across the concept and signal extraction tiers so
that both components make consistent decisions when selecting domain-specific
resources such as lexicons or spaCy models.
"""

from __future__ import annotations

from typing import Optional

from app.models.paper import Paper


def infer_domain_key(paper: Optional[Paper]) -> Optional[str]:
    """Return a coarse domain label inferred from the paper metadata.

    The heuristics intentionally remain lightweight – they only look at readily
    available metadata such as the title and venue so that the function can run
    early in the pipeline before any heavy parsing occurs.  The returned key is
    used to select domain-tuned lexicons or NLP models.  `None` is returned when
    the heuristics cannot confidently assign a domain.
    """

    if paper is None:
        return None

    fields = [paper.venue, paper.file_content_type, paper.title]
    combined = " ".join(filter(None, (value or "" for value in fields))).strip()
    if not combined:
        return None

    lowered = combined.lower()

    biology_cues = (
        "biology",
        "biochem",
        "genom",
        "microbio",
        "immun",
        "cell",
        "organism",
        "protein",
        "enzyme",
        "med",
    )
    if any(cue in lowered for cue in biology_cues):
        return "biology"

    materials_cues = (
        "material",
        "materials",
        "alloy",
        "polymer",
        "ceramic",
        "perovskite",
        "graphene",
        "nanotube",
        "battery",
        "cathode",
        "anode",
        "electrode",
        "composite",
        "crystal",
        "oxide",
    )
    if any(cue in lowered for cue in materials_cues):
        return "materials"

    translation_cues = (
        "translation",
        "machine translation",
        "mt workshop",
        "wmt",
        "multilingual benchmark",
    )
    if any(cue in lowered for cue in translation_cues):
        return "machine_translation"

    return None

