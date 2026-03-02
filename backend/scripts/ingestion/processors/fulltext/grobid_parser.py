# backend/scripts/ingestion/processors/fulltext/grobid_parser.py

"""
GROBID PDF parser — extracts structured sections from academic papers.

Sends PDFs to a local GROBID server (Docker), parses the TEI XML
response into clean section dicts.

GROBID setup (run once):
  docker pull grobid/grobid:0.8.1
  docker run --rm -p 8070:8070 grobid/grobid:0.8.1

Output format per paper:
  {
    "paper_id": "arxiv:2504.12345",
    "grobid_title": "...",
    "sections": [
      {"heading": "Introduction", "text": "...", "section_type": "introduction"},
      {"heading": "3.1 Encoder", "text": "...", "section_type": "method"},
      ...
    ]
  }
"""

import re
import requests
from typing import List, Dict, Optional
from pathlib import Path
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed


# Section heading → type mapping
# We only keep high-value sections and skip noise
_SECTION_TYPE_MAP = {
    # KEEP — high retrieval value
    "abstract":         "abstract",
    "introduction":     "introduction",
    "method":           "method",
    "methodology":      "method",
    "approach":         "method",
    "framework":        "method",
    "architecture":     "method",
    "model":            "method",
    "system":           "method",
    "algorithm":        "method",
    "implementation":   "method",
    "training":         "method",
    "formulation":      "method",
    "proposed":         "method",
    "design":           "method",
    "pipeline":         "method",
    "procedure":        "method",
    "technique":        "method",
    "strategy":         "method",
    "network":          "method",
    "representation":   "method",
    "feature":          "method",
    "features":         "method",
    "learning":         "method",
    "objective":        "method",
    "loss":             "method",
    "inference":        "method",
    "decoding":         "method",
    "encoding":         "method",
    "attention":        "method",
    "mechanism":        "method",
    "module":           "method",
    "component":        "method",
    "optimization":     "method",
    "sampling":         "method",
    "generation":       "method",
    "fine-tuning":      "method",
    "pre-training":     "method",
    "pretraining":      "method",
    "prompt":           "method",
    "retrieval":        "method",
    "experiment":       "experiment",
    "experiments":      "experiment",
    "experimental":     "experiment",
    "evaluation":       "experiment",
    "results":          "experiment",
    "analysis":         "experiment",
    "ablation":         "experiment",
    "benchmark":        "experiment",
    "benchmarks":       "experiment",
    "setup":            "experiment",
    "setting":          "experiment",
    "settings":         "experiment",
    "baseline":         "experiment",
    "baselines":        "experiment",
    "comparison":       "experiment",
    "performance":      "experiment",
    "quantitative":     "experiment",
    "qualitative":      "experiment",
    "dataset":          "experiment",
    "datasets":         "experiment",
    "data":             "experiment",
    "metric":           "experiment",
    "metrics":          "experiment",
    "hyperparameter":   "experiment",
    "case study":       "experiment",
    "case studies":     "experiment",
    "user study":       "experiment",
    "human evaluation":  "experiment",
    "discussion":       "discussion",
    "limitation":       "discussion",
    "limitations":      "discussion",
    "future work":      "discussion",
    "future direction":  "discussion",
    "implication":      "discussion",
    "implications":     "discussion",
    "challenge":        "discussion",
    "challenges":       "discussion",
    "conclusion":       "conclusion",
    "conclusions":      "conclusion",
    "concluding":       "conclusion",
    "summary":          "conclusion",
    "closing":          "conclusion",
    "background":       "background",
    "preliminary":      "background",
    "preliminaries":    "background",
    "problem":          "background",
    "motivation":       "background",
    "overview":         "background",
    "definition":       "background",
    "definitions":      "background",
    "notation":         "background",
    "formalism":        "background",
    "task":             "background",
    "setting":          "background",
    "scenario":         "background",

    # SKIP — low retrieval value / noise
    "related work":     "related_work",
    "related works":    "related_work",
    "prior work":       "related_work",
    "literature":       "related_work",
    "acknowledgment":   "skip",
    "acknowledgments":  "skip",
    "acknowledgement":  "skip",
    "acknowledgements": "skip",
    "references":       "skip",
    "bibliography":     "skip",
    "appendix":         "skip",
    "supplementary":    "skip",
    "ethics":           "skip",
    "broader impact":   "skip",
    "funding":          "skip",
    "author contributions": "skip",
    "data availability": "skip",
    "conflict of interest": "skip",
}

# Sections to skip entirely during chunking
_SKIP_TYPES = {"skip", "related_work"}


def _classify_section(heading: str, position_ratio: float = 0.5) -> str:
    """
    Classify a section heading into a type.
    Uses fuzzy matching against known patterns, with a position-based
    fallback for unrecognised headings.

    Args:
        heading: Section heading text
        position_ratio: Where this section sits in the paper (0.0 = start, 1.0 = end)
    """
    if not heading:
        return "unknown"

    h = heading.lower().strip()

    # Remove numbering: "3.1 Encoder Architecture" → "encoder architecture"
    h_clean = re.sub(r'^[\d\.\s]+', '', h).strip()

    # Direct match
    if h_clean in _SECTION_TYPE_MAP:
        return _SECTION_TYPE_MAP[h_clean]

    # Partial match (check if any key is contained in the heading)
    for key, stype in _SECTION_TYPE_MAP.items():
        if key in h_clean:
            return stype

    # Position-based fallback: academic papers follow IMRAD structure
    # (Introduction, Methods, Results, And Discussion) almost universally.
    # Not perfect for unusual papers, but better than an unlabeled "other".
    if position_ratio < 0.15:
        return "introduction"
    elif position_ratio < 0.35:
        return "method"
    elif position_ratio < 0.75:
        return "experiment"
    elif position_ratio < 0.90:
        return "discussion"
    else:
        return "conclusion"


class GrobidParser:
    """
    Parse PDFs into structured sections using GROBID.

    Requires GROBID running locally via Docker:
      docker run --rm -p 8070:8070 grobid/grobid:0.8.1
    """

    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        timeout: int = 120,
    ):
        self.grobid_url = grobid_url.rstrip("/")
        self.timeout = timeout
        self._check_server()

    def _check_server(self):
        """Verify GROBID is running."""
        try:
            r = requests.get(f"{self.grobid_url}/api/isalive", timeout=10)
            if r.status_code == 200:
                logger.info(f"GROBID server alive at {self.grobid_url}")
            else:
                logger.warning(f"GROBID returned status {r.status_code}")
        except requests.ConnectionError:
            logger.error(
                f"Cannot connect to GROBID at {self.grobid_url}. "
                f"Start it with: docker run --rm -p 8070:8070 grobid/grobid:0.8.1"
            )
            raise ConnectionError(f"GROBID not available at {self.grobid_url}")

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(10))
    def _parse_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Send a PDF to GROBID and get back TEI XML.
        Returns raw XML string or None on failure.
        """
        # Skip very large PDFs that crash GROBID
        file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 20:
            logger.warning(f"Skipping oversized PDF ({file_size_mb:.1f}MB): {pdf_path}")
            return None

        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files={"input": f},
                data={
                    "segmentSentences": "1",
                    "includeRawCitations": "0",
                    "includeRawAffiliations": "0",
                },
                timeout=self.timeout,
            )

        if response.status_code != 200:
            logger.warning(f"GROBID error {response.status_code} for {pdf_path}")
            return None

        return response.text

    def _parse_tei_xml(self, xml_text: str, paper_id: str) -> Dict:
        """
        Parse GROBID TEI XML into structured sections.

        Returns:
          {
            "paper_id": "...",
            "grobid_title": "...",
            "abstract_text": "...",
            "sections": [
              {"heading": "...", "text": "...", "section_type": "method"},
              ...
            ]
          }
        """
        soup = BeautifulSoup(xml_text, "xml")
        result = {
            "paper_id": paper_id,
            "grobid_title": "",
            "abstract_text": "",
            "sections": [],
        }

        # Extract title
        title_tag = soup.find("title", attrs={"type": "main"})
        if title_tag:
            result["grobid_title"] = title_tag.get_text(strip=True)

        # Extract abstract
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            result["abstract_text"] = abstract_tag.get_text(" ", strip=True)

        # Extract body sections
        body = soup.find("body")
        if not body:
            logger.warning(f"No body found in GROBID output for {paper_id}")
            return result

        # Walk through <div> elements (GROBID's section containers)
        all_divs = body.find_all("div", recursive=False)
        total_divs = max(len(all_divs), 1)

        for div_idx, div in enumerate(all_divs):
            position_ratio = div_idx / total_divs
            head = div.find("head")
            heading = head.get_text(strip=True) if head else ""

            # Collect all paragraph text in this div
            paragraphs = []
            for p in div.find_all("p"):
                text = p.get_text(" ", strip=True)
                if text:
                    paragraphs.append(text)

            # Also handle nested divs (subsections)
            sub_divs = div.find_all("div", recursive=True)
            total_sub = max(len(sub_divs), 1)
            for sub_idx, sub_div in enumerate(sub_divs):
                # Subsection position: interpolate between parent div's range
                sub_ratio = position_ratio + (sub_idx / total_sub) * (1.0 / total_divs)
                sub_head = sub_div.find("head")
                sub_heading = sub_head.get_text(strip=True) if sub_head else ""

                sub_paragraphs = []
                for p in sub_div.find_all("p"):
                    text = p.get_text(" ", strip=True)
                    if text and text not in paragraphs:  # avoid duplicates
                        sub_paragraphs.append(text)

                if sub_paragraphs:
                    section_type = _classify_section(sub_heading or heading, sub_ratio)
                    result["sections"].append({
                        "heading": sub_heading or heading,
                        "text": "\n\n".join(sub_paragraphs),
                        "section_type": section_type,
                    })

            # Add the parent div's direct paragraphs if any
            if paragraphs:
                section_type = _classify_section(heading, position_ratio)
                result["sections"].append({
                    "heading": heading,
                    "text": "\n\n".join(paragraphs),
                    "section_type": section_type,
                })

        # Deduplicate sections (GROBID sometimes creates overlaps)
        seen_texts = set()
        unique_sections = []
        for sec in result["sections"]:
            # Use first 200 chars as fingerprint
            fp = sec["text"][:200]
            if fp not in seen_texts:
                seen_texts.add(fp)
                unique_sections.append(sec)
        result["sections"] = unique_sections

        return result

    def parse_paper(self, paper: Dict) -> Optional[Dict]:
        """
        Parse a single paper's PDF into structured sections.

        Args:
            paper: Dict with at least 'id' and 'pdf_local_path' fields.

        Returns:
            Parsed result dict or None on failure.
        """
        pdf_path = paper.get("pdf_local_path")
        if not pdf_path or not Path(pdf_path).exists():
            logger.warning(f"No PDF found for {paper['id']}")
            return None

        xml = self._parse_pdf(pdf_path)
        if not xml:
            return None

        result = self._parse_tei_xml(xml, paper["id"])

        # Filter out skip sections
        result["sections"] = [
            s for s in result["sections"]
            if s["section_type"] not in _SKIP_TYPES
        ]

        section_count = len(result["sections"])
        total_chars = sum(len(s["text"]) for s in result["sections"])

        logger.debug(
            f"{paper['id']}: {section_count} sections, "
            f"{total_chars:,} chars extracted"
        )

        return result

    def _wait_for_grobid(self, max_wait: int = 120):
        """Wait for GROBID to be alive (e.g. after a crash/restart)."""
        import time
        for i in range(max_wait // 5):
            try:
                r = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            if i == 0:
                logger.warning("GROBID not responding, waiting for it to recover...")
            time.sleep(5)
        logger.error(f"GROBID did not recover after {max_wait}s")
        return False

    def parse_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Parse multiple papers. Returns list of successfully parsed results.
        """
        results = []
        failed = 0
        consecutive_failures = 0

        for paper in papers:
            try:
                parsed = self.parse_paper(paper)
                if parsed and parsed["sections"]:
                    results.append(parsed)
                    consecutive_failures = 0
                else:
                    failed += 1
                    consecutive_failures += 1
            except Exception as e:
                logger.warning(f"Parse failed for {paper['id']}: {e}")
                failed += 1
                consecutive_failures += 1

                # If 3+ failures in a row, GROBID probably crashed
                if consecutive_failures >= 3:
                    logger.warning("Multiple consecutive failures — checking GROBID health...")
                    if not self._wait_for_grobid():
                        logger.error("GROBID is down. Returning partial results.")
                        break
                    consecutive_failures = 0

        logger.info(f"GROBID parsing: {len(results)} OK, {failed} failed")
        return results
