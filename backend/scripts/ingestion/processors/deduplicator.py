# backend/scripts/ingestion/processors/deduplicator.py

import re
from typing import List, Dict
from loguru import logger

class Deduplicator:
    """Remove duplicate papers"""
    
    def deduplicate(
        self,
        papers: List[Dict],
        keep_latest: bool = True
    ) -> List[Dict]:
        """
        Remove duplicate papers
        
        Args:
            papers: List of papers
            keep_latest: Keep the latest version if duplicates found
        
        Returns:
            List of unique papers
        """
        logger.info(f"Deduplicating {len(papers)} papers...")
        
        paper_dict = {}
        
        for paper in papers:
            paper_id = paper['id']
            
            # Normalize ID (remove version)
            base_id = self._normalize_id(paper_id)
            
            if base_id not in paper_dict:
                paper_dict[base_id] = paper
            else:
                # If duplicate, keep the latest version
                if keep_latest:
                    existing_version = self._extract_version(paper_dict[base_id]['id'])
                    new_version = self._extract_version(paper_id)
                    
                    if new_version > existing_version:
                        paper_dict[base_id] = paper
                        logger.debug(f"Replaced {existing_version} with {new_version} for {base_id}")
        
        unique_papers = list(paper_dict.values())
        
        duplicates_removed = len(papers) - len(unique_papers)
        
        logger.info(f"Unique papers: {len(unique_papers)}")
        logger.info(f"Removed {duplicates_removed} duplicates")
        
        return unique_papers
    
    def _normalize_id(self, paper_id: str) -> str:
        """
        Normalize paper ID (remove version)
        
        Examples:
            'arxiv:2301.12345v1' -> 'arxiv:2301.12345'
            'arxiv:2301.12345v2' -> 'arxiv:2301.12345'
            'arxiv:2301.12345' -> 'arxiv:2301.12345'
        """
        # Check for version pattern (v followed by digits at the end)
        # Pattern: v followed by one or more digits at the end of string
        version_pattern = r'v\d+$'
        if re.search(version_pattern, paper_id):
            # Remove version suffix
            return re.sub(version_pattern, '', paper_id)
        return paper_id
    
    def _extract_version(self, paper_id: str) -> int:
        """
        Extract version number from paper ID
        
        Examples:
            'arxiv:2301.12345v1' -> 1
            'arxiv:2301.12345v2' -> 2
            'arxiv:2301.12345' -> 0
        """
        if 'v' in paper_id:
            try:
                version = paper_id.split('v')[-1]
                return int(version)
            except ValueError:
                return 0
        return 0