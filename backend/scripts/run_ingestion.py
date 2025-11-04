# backend/scripts/run_ingestion.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
from loguru import logger
from backend.scripts.ingestion.pipeline import IngestionPipeline
from backend.scripts.ingestion.categories import PRESETS, ALL_CS_AND_ENG_CATEGORIES

def main():
    """
    Main entry point for ingestion
    
    Usage:
        python backend/scripts/run_ingestion.py
        python backend/scripts/run_ingestion.py --fresh
        python backend/scripts/run_ingestion.py --categories cs.AI cs.CL --papers 2000
        python backend/scripts/run_ingestion.py --preset comprehensive --papers 5000
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Ingest papers into Pinecone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default categories (cs.AI, cs.CL, cs.LG)
  python backend/scripts/run_ingestion.py --papers 1000
  
  # Use specific categories
  python backend/scripts/run_ingestion.py --categories cs.AI cs.CV cs.LG --papers 2000
  
  # Use preset category groups
  python backend/scripts/run_ingestion.py --preset cs-all --papers 5000
  python backend/scripts/run_ingestion.py --preset comprehensive --papers 3000
  
Available presets:
  cs-core: Core AI/ML categories (cs.AI, cs.CL, cs.LG, cs.CV, cs.NE, cs.RO)
  cs-systems: Systems and architecture categories
  cs-data: Data and database categories
  cs-security: Security and networks
  cs-graphics: Graphics and multimedia
  cs-all: All Computer Science categories (~40 categories)
  eess-all: Electrical Engineering categories
  biotech: Quantitative Biology/Bioinformatics (~10 categories)
  quant-finance: Computational Finance categories
  ai-applications: AI/ML applications (Stats, Finance, Economics, Physics)
  comprehensive: All CS + Engineering + Biotech + Finance + Math/Stats (~75+ categories)
        """
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh (ignore checkpoints)'
    )
    
    category_group = parser.add_mutually_exclusive_group()
    
    category_group.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='arXiv categories to fetch (e.g., cs.AI cs.CL cs.LG)'
    )
    
    category_group.add_argument(
        '--preset',
        choices=list(PRESETS.keys()),
        default=None,
        help='Use a preset category group (see available presets below)'
    )
    
    parser.add_argument(
        '--papers',
        type=int,
        default=1000,
        help='Papers per category (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Determine categories to use
    if args.preset:
        categories = PRESETS[args.preset]
        logger.info(f"Using preset '{args.preset}' with {len(categories)} categories")
    elif args.categories:
        categories = args.categories
    else:
        # Default to comprehensive preset for more papers
        categories = PRESETS['comprehensive']
        logger.info(f"Using default comprehensive preset with {len(categories)} categories")
    
    # Configure logging
    logger.add(
        "backend/data/logs/ingestion_{time}.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO"
    )
    
    logger.info("Starting ingestion pipeline...")
    logger.info(f"Categories ({len(categories)}): {categories[:10]}{'...' if len(categories) > 10 else ''}")
    logger.info(f"Papers per category: {args.papers}")
    logger.info(f"Total target papers: {len(categories) * args.papers:,}")
    logger.info(f"Resume mode: {not args.fresh}")
    
    # Run pipeline
    pipeline = IngestionPipeline(resume=not args.fresh)
    
    try:
        result = pipeline.run(
            categories=categories,
            papers_per_category=args.papers
        )
        
        logger.info("Pipeline completed successfully!")
        return result
        
    except KeyboardInterrupt:
        logger.warning("\n Interrupted by user")
        logger.info("Run again to resume from checkpoint")
        return None
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.info("Run again to resume from checkpoint")
        raise

if __name__ == "__main__":
    main()