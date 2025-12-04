"""
Data Pipeline for Bio-Link Agent

This script handles:
1. Fetching clinical trials from ClinicalTrials.gov
2. Fetching literature from PubMed
3. Indexing trials into vector database
4. Caching data for faster subsequent runs
"""

import sys
import os
import json
import pickle
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.pubmed_server import PubMedServer
from mcp_servers.clinicaltrials_server import ClinicalTrialsServer
from utils.embeddings import EmbeddingManager
from loguru import logger
from tqdm import tqdm


# Configuration (DOUBLED data size for better coverage)
CONDITIONS = [
    {"name": "lung_cancer", "query": "lung cancer", "trials": 200, "papers": 60},
    {"name": "glioblastoma", "query": "glioblastoma", "trials": 200, "papers": 60},
    {"name": "type_2_diabetes", "query": "type 2 diabetes", "trials": 200, "papers": 60},
]

DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
TRIALS_DIR = DATA_DIR / "trials"
PAPERS_DIR = DATA_DIR / "papers"


def setup_directories():
    """Create necessary directories."""
    for dir_path in [DATA_DIR, CACHE_DIR, TRIALS_DIR, PAPERS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("Directories created")


def fetch_trials(condition: dict, server: ClinicalTrialsServer) -> list:
    """Fetch trials for a condition."""
    cache_file = TRIALS_DIR / f"{condition['name']}_trials.json"
    
    # Check cache
    if cache_file.exists():
        logger.info(f"Loading cached trials for {condition['name']}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Fetch from API
    logger.info(f"Fetching trials for: {condition['query']}")
    trials = server.search_and_fetch(
        query=condition['query'],
        max_results=condition['trials'],
        status="RECRUITING"
    )
    
    # Also fetch completed trials for more data
    logger.info(f"Fetching completed trials for: {condition['query']}")
    completed_trials = server.search_and_fetch(
        query=condition['query'],
        max_results=condition['trials'] // 2,
        status="COMPLETED"
    )
    
    all_trials = trials + completed_trials
    
    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    
    logger.info(f"Cached {len(all_trials)} trials for {condition['name']}")
    return all_trials


def fetch_papers(condition: dict, server: PubMedServer) -> list:
    """Fetch papers for a condition."""
    cache_file = PAPERS_DIR / f"{condition['name']}_papers.json"
    
    # Check cache
    if cache_file.exists():
        logger.info(f"Loading cached papers for {condition['name']}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Fetch from API
    logger.info(f"Fetching papers for: {condition['query']}")
    papers = server.search_and_fetch(
        query=condition['query'],
        max_results=condition['papers']
    )
    
    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(papers, f, indent=2)
    
    logger.info(f"Cached {len(papers)} papers for {condition['name']}")
    return papers


def index_trials(trials: list, condition_name: str, embeddings: EmbeddingManager):
    """Index trials into vector database."""
    logger.info(f"Indexing {len(trials)} trials for {condition_name}")
    embeddings.index_trials(
        trials=trials,
        collection_name=condition_name,
        overwrite=True
    )
    logger.info(f"Indexing complete for {condition_name}")


def generate_statistics(all_data: dict) -> dict:
    """Generate statistics about the collected data."""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "conditions": {}
    }
    
    for condition_name, data in all_data.items():
        trials = data.get("trials", [])
        papers = data.get("papers", [])
        
        # Trial statistics
        trial_stats = {
            "total_trials": len(trials),
            "recruiting": sum(1 for t in trials if t.get("status") == "RECRUITING"),
            "completed": sum(1 for t in trials if t.get("status") == "COMPLETED"),
            "interventions": set(),
            "phases": set(),
        }
        
        for trial in trials:
            for intervention in trial.get("interventions", []):
                trial_stats["interventions"].add(intervention.get("name", ""))
            for phase in trial.get("phases", []):
                trial_stats["phases"].add(phase)
        
        trial_stats["interventions"] = list(trial_stats["interventions"])[:20]  # Top 20
        trial_stats["phases"] = list(trial_stats["phases"])
        trial_stats["unique_interventions"] = len(trial_stats["interventions"])
        
        # Paper statistics
        paper_stats = {
            "total_papers": len(papers),
            "years": {},
            "journals": set(),
        }
        
        for paper in papers:
            year = paper.get("year", "Unknown")
            paper_stats["years"][year] = paper_stats["years"].get(year, 0) + 1
            paper_stats["journals"].add(paper.get("journal", "Unknown"))
        
        paper_stats["unique_journals"] = len(paper_stats["journals"])
        paper_stats["journals"] = list(paper_stats["journals"])[:10]  # Top 10
        
        stats["conditions"][condition_name] = {
            "trials": trial_stats,
            "papers": paper_stats
        }
    
    return stats


def main(auto_run: bool = False):
    """Main data pipeline."""
    print("\n" + "="*70)
    print("BIO-LINK AGENT - DATA PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Fetch clinical trials from ClinicalTrials.gov")
    print("  2. Fetch literature from PubMed")
    print("  3. Index trials into vector database")
    print("  4. Cache all data for faster subsequent runs")
    print("\nConditions to process:")
    for cond in CONDITIONS:
        print(f"  - {cond['name']}: {cond['trials']} trials, {cond['papers']} papers")
    print("\nEstimated time: 5-10 minutes")
    print("="*70 + "\n")
    
    if not auto_run:
        input("Press Enter to start data collection...")
    else:
        print("Starting data collection (auto-run mode)...")
    
    # Setup
    setup_directories()
    
    # Initialize servers
    pubmed = PubMedServer()
    clinicaltrials = ClinicalTrialsServer()
    embeddings = EmbeddingManager()
    
    all_data = {}
    
    # Process each condition
    for i, condition in enumerate(CONDITIONS, 1):
        print(f"\n{'='*70}")
        print(f"PROCESSING {i}/{len(CONDITIONS)}: {condition['name'].upper()}")
        print("="*70)
        
        # Fetch trials
        print("\n[Step 1/3] Fetching clinical trials...")
        trials = fetch_trials(condition, clinicaltrials)
        print(f"  Fetched {len(trials)} trials")
        
        # Fetch papers
        print("\n[Step 2/3] Fetching literature...")
        papers = fetch_papers(condition, pubmed)
        print(f"  Fetched {len(papers)} papers")
        
        # Index trials
        print("\n[Step 3/3] Indexing trials into vector database...")
        index_trials(trials, condition['name'], embeddings)
        print(f"  Indexed {len(trials)} trials")
        
        all_data[condition['name']] = {
            "trials": trials,
            "papers": papers
        }
        
        print(f"\n✓ {condition['name']} complete!")
    
    # Generate and save statistics
    print("\n" + "="*70)
    print("GENERATING STATISTICS")
    print("="*70)
    
    stats = generate_statistics(all_data)
    
    stats_file = DATA_DIR / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE")
    print("="*70)
    
    total_trials = sum(len(d['trials']) for d in all_data.values())
    total_papers = sum(len(d['papers']) for d in all_data.values())
    
    print(f"\nSummary:")
    print(f"  Total Trials Collected: {total_trials}")
    print(f"  Total Papers Collected: {total_papers}")
    print(f"  Conditions: {len(CONDITIONS)}")
    
    print(f"\nData saved to:")
    print(f"  - Trials: {TRIALS_DIR}/")
    print(f"  - Papers: {PAPERS_DIR}/")
    print(f"  - Vector DB: {DATA_DIR / 'indexed_trials'}/")
    print(f"  - Statistics: {stats_file}")
    
    print("\nPer-condition breakdown:")
    for cond_name, cond_stats in stats['conditions'].items():
        t = cond_stats['trials']
        p = cond_stats['papers']
        print(f"\n  {cond_name}:")
        print(f"    Trials: {t['total_trials']} ({t['recruiting']} recruiting, {t['completed']} completed)")
        print(f"    Papers: {p['total_papers']} from {p['unique_journals']} journals")
        print(f"    Unique interventions: {t['unique_interventions']}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Run the semantic matcher demo:")
    print("   python demos/demo_matcher.py")
    print("\n2. Run the landscape agent demo:")
    print("   python demos/demo_landscape.py")
    print("\n3. Or use the agents programmatically:")
    print("   from agents.semantic_matcher import SemanticMatcher")
    print("   from agents.landscape_agent import TherapeuticLandscapeAgent")
    print("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bio-Link Agent Data Pipeline")
    parser.add_argument("--auto", action="store_true", help="Run without prompts")
    args = parser.parse_args()
    main(auto_run=args.auto)

