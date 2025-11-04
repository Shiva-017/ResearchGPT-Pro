# backend/scripts/ingestion/processors/validator.py

from typing import List, Dict
from pydantic import BaseModel, field_validator, Field
from datetime import datetime
from loguru import logger

class PaperSchema(BaseModel):
    """Validation schema for papers"""
    
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=10, max_length=1000)
    abstract: str = Field(..., min_length=50)
    authors: List[str] = Field(..., min_length=1)
    year: int = Field(..., ge=1990, le=datetime.now().year + 1)
    published: str
    categories: List[str] = Field(default_factory=list)
    
    @field_validator('title')
    @classmethod
    def title_valid(cls, v):
        """Ensure title is meaningful"""
        if not v or v.isspace():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @field_validator('abstract')
    @classmethod
    def abstract_meaningful(cls, v):
        """Ensure abstract has content"""
        cleaned = v.strip()
        if len(cleaned) < 50:
            raise ValueError('Abstract too short (< 50 chars)')
        return cleaned
    
    @field_validator('authors')
    @classmethod
    def authors_valid(cls, v):
        """Clean and validate authors"""
        if not v:
            raise ValueError('Must have at least one author')
        
        # Clean author names
        cleaned = [a.strip() for a in v if a and a.strip()]
        
        if not cleaned:
            raise ValueError('No valid authors after cleaning')
        
        return cleaned

class PaperValidator:
    """Validate and clean papers"""
    
    def __init__(self):
        self.validation_errors = []
    
    def validate_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Validate list of papers
        
        Args:
            papers: Raw paper dictionaries
        
        Returns:
            List of valid papers
        """
        logger.info(f"Validating {len(papers)} papers...")
        
        valid_papers = []
        invalid_count = 0
        
        for paper in papers:
            try:
                # Validate using Pydantic
                validated = PaperSchema(**paper)
                
                # Convert back to dict (Pydantic v2 uses model_dump)
                valid_paper = validated.model_dump()
                
                # Add fields that weren't in schema
                valid_paper.update({
                    k: v for k, v in paper.items()
                    if k not in valid_paper
                })
                
                valid_papers.append(valid_paper)
                
            except Exception as e:
                invalid_count += 1
                self.validation_errors.append({
                    'paper_id': paper.get('id', 'unknown'),
                    'error': str(e)
                })
                
                if invalid_count <= 5:  # Show first 5 errors
                    logger.warning(f"Invalid paper {paper.get('id')}: {e}")
        
        logger.info(f"Valid papers: {len(valid_papers)}")
        logger.info(f"Invalid papers: {invalid_count}")
        
        if invalid_count > 5:
            logger.info(f"   (showing first 5 errors, {invalid_count - 5} more)")
        
        return valid_papers
    
    def get_validation_report(self) -> Dict:
        """Get validation error report"""
        return {
            'total_errors': len(self.validation_errors),
            'errors': self.validation_errors
        }