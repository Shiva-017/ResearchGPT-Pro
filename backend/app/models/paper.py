# backend/app/models/paper.py

"""
Pydantic models for paper data.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Paper(BaseModel):
    """Core paper model."""
    id: str
    arxiv_id: str = ""
    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    year: int
    published: str = ""
    categories: List[str] = Field(default_factory=list)
    primary_category: str = ""
    pdf_url: str = ""
    source: str = "arxiv"
    citation_count: int = 0


class PaperDetail(Paper):
    """Extended paper model with extra fields."""
    field: str = ""
    has_code: bool = False
    is_survey: bool = False
    author_count: int = 0
    citation_tier: str = "low"
