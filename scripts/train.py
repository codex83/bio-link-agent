"""
Training Script for Bio-Link Agent

This script pre-processes the collected data:
1. Extracts entities from all cached papers using Ollama
2. Extracts relationships between entities
3. Builds and saves knowledge graphs for each condition
4. Caches all extracted data for faster inference

This is the "training" phase - preparing the knowledge base.
"""

import sys
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from collections import Counter

# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_client import OllamaClient
from utils.graph_builder import KnowledgeGraphBuilder
from loguru import logger
from tqdm import tqdm
import time


# Configuration
DATA_DIR = Path("./data")
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPHS_DIR = DATA_DIR / "graphs"

CONDITIONS = ["lung_cancer", "glioblastoma", "type_2_diabetes"]


def setup_directories():
    """Create necessary directories."""
    for dir_path in [PROCESSED_DIR, GRAPHS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("Directories ready")


def load_papers(condition: str) -> list:
    """Load cached papers for a condition."""
    papers_file = PAPERS_DIR / f"{condition}_papers.json"
    
    if not papers_file.exists():
        logger.error(f"Papers file not found: {papers_file}")
        logger.error("Run data_pipeline.py first to collect data")
        return []
    
    with open(papers_file, 'r') as f:
        papers = json.load(f)
    
    logger.info(f"Loaded {len(papers)} papers for {condition}")
    return papers


def extract_entities_from_papers(papers: list, llm: OllamaClient, condition: str) -> list:
    """Extract entities from all papers."""
    processed_file = PROCESSED_DIR / f"{condition}_entities.json"
    
    # Check if already processed
    if processed_file.exists():
        logger.info(f"Loading cached entities for {condition}")
        with open(processed_file, 'r') as f:
            return json.load(f)
    
    logger.info(f"Extracting entities from {len(papers)} papers...")
    
    processed_papers = []
    
    for i, paper in enumerate(tqdm(papers, desc=f"Processing {condition}")):
        abstract = paper.get("abstract", "")
        
        if not abstract or len(abstract) < 50:
            logger.debug(f"Skipping paper {paper.get('pmid', 'unknown')} - no/short abstract")
            continue
        
        try:
            # Extract entities
            entities = llm.extract_entities(abstract)
            
            # Extract relationships
            relationships = llm.extract_relationships(abstract, entities)
            
            processed_papers.append({
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "abstract": abstract,
                "entities": entities,
                "relationships": relationships,
            })
            
            # Small delay to prevent overwhelming Ollama
            time.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Error processing paper {paper.get('pmid', 'unknown')}: {e}")
            continue
    
    # Cache results
    with open(processed_file, 'w') as f:
        json.dump(processed_papers, f, indent=2)
    
    logger.info(f"Extracted entities from {len(processed_papers)} papers")
    return processed_papers


def build_knowledge_graph(processed_papers: list, condition: str) -> dict:
    """Build knowledge graph from processed papers."""
    logger.info(f"Building knowledge graph for {condition}")
    
    builder = KnowledgeGraphBuilder()
    
    # Add all entities and relationships
    for paper in processed_papers:
        builder.add_entities(paper.get("entities", {}))
        builder.add_relationships(paper.get("relationships", []))
    
    # Get graph summary
    summary = builder.get_summary()
    logger.info(f"Graph for {condition}: {summary}")
    
    # Save graph visualization
    graph_image = GRAPHS_DIR / f"{condition}_knowledge_graph.png"
    builder.visualize(str(graph_image))
    logger.info(f"Graph visualization saved to {graph_image}")
    
    # Save graph as pickle for later use
    graph_file = GRAPHS_DIR / f"{condition}_graph.pkl"
    with open(graph_file, 'wb') as f:
        pickle.dump(builder.graph, f)
    logger.info(f"Graph saved to {graph_file}")
    
    # Aggregate entity statistics
    all_drugs = []
    all_targets = []
    all_pathways = []
    all_outcomes = []
    
    for paper in processed_papers:
        entities = paper.get("entities", {})
        all_drugs.extend(entities.get("drugs", []))
        all_targets.extend(entities.get("targets", []))
        all_pathways.extend(entities.get("pathways", []))
        all_outcomes.extend(entities.get("outcomes", []))
    
    return {
        "condition": condition,
        "papers_processed": len(processed_papers),
        "graph_summary": summary,
        "top_drugs": Counter(all_drugs).most_common(10),
        "top_targets": Counter(all_targets).most_common(10),
        "top_pathways": Counter(all_pathways).most_common(10),
        "top_outcomes": Counter(all_outcomes).most_common(10),
    }


def generate_training_report(all_stats: list):
    """Generate a training summary report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "conditions": {}
    }
    
    for stats in all_stats:
        condition = stats["condition"]
        report["conditions"][condition] = {
            "papers_processed": stats["papers_processed"],
            "graph_nodes": stats["graph_summary"]["num_nodes"],
            "graph_edges": stats["graph_summary"]["num_edges"],
            "top_drugs": [{"name": d, "count": c} for d, c in stats["top_drugs"]],
            "top_targets": [{"name": t, "count": c} for t, c in stats["top_targets"]],
            "top_pathways": [{"name": p, "count": c} for p, c in stats["top_pathways"]],
            "top_outcomes": [{"name": o, "count": c} for o, c in stats["top_outcomes"]],
        }
    
    # Save report
    report_file = DATA_DIR / "training_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("BIO-LINK AGENT - TRAINING PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Load cached papers from data collection")
    print("  2. Extract entities (drugs, targets, pathways, outcomes) using Ollama")
    print("  3. Extract relationships between entities")
    print("  4. Build knowledge graphs for each condition")
    print("  5. Save processed data for faster inference")
    print("\nConditions to process:")
    for cond in CONDITIONS:
        print(f"  - {cond}")
    print("\nNote: This uses Ollama for LLM inference. May take 10-20 minutes.")
    print("="*70 + "\n")
    
    # Check if papers exist
    missing = []
    for cond in CONDITIONS:
        if not (PAPERS_DIR / f"{cond}_papers.json").exists():
            missing.append(cond)
    
    if missing:
        print(f"ERROR: Missing paper data for: {missing}")
        print("Run 'python data_pipeline.py --auto' first")
        return
    
    print("Starting training...\n")
    
    # Setup
    setup_directories()
    
    # Initialize LLM client
    print("Initializing Ollama client...")
    llm = OllamaClient(model="gemma3:12b", temperature=0.1)
    
    all_stats = []
    
    # Process each condition
    for i, condition in enumerate(CONDITIONS, 1):
        print(f"\n{'='*70}")
        print(f"TRAINING {i}/{len(CONDITIONS)}: {condition.upper()}")
        print("="*70)
        
        # Load papers
        print("\n[Step 1/3] Loading papers...")
        papers = load_papers(condition)
        
        if not papers:
            print(f"  No papers found for {condition}, skipping...")
            continue
        
        print(f"  Loaded {len(papers)} papers")
        
        # Extract entities
        print("\n[Step 2/3] Extracting entities with Ollama...")
        print("  This may take several minutes per condition...")
        processed_papers = extract_entities_from_papers(papers, llm, condition)
        
        # Build knowledge graph
        print("\n[Step 3/3] Building knowledge graph...")
        stats = build_knowledge_graph(processed_papers, condition)
        all_stats.append(stats)
        
        # Print condition summary
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE: {condition.upper()}")
        print("="*70)
        print(f"  Papers processed: {stats['papers_processed']}")
        print(f"  Graph nodes: {stats['graph_summary']['num_nodes']}")
        print(f"  Graph edges: {stats['graph_summary']['num_edges']}")
        print(f"\n  Top Drugs: {', '.join([d for d, c in stats['top_drugs'][:5]])}")
        print(f"  Top Targets: {', '.join([t for t, c in stats['top_targets'][:5]])}")
        print(f"  Top Pathways: {', '.join([p for p, c in stats['top_pathways'][:5]])}")
    
    # Generate final report
    print("\n" + "="*70)
    print("GENERATING TRAINING REPORT")
    print("="*70)
    
    report = generate_training_report(all_stats)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    total_papers = sum(s["papers_processed"] for s in all_stats)
    total_nodes = sum(s["graph_summary"]["num_nodes"] for s in all_stats)
    total_edges = sum(s["graph_summary"]["num_edges"] for s in all_stats)
    
    print(f"\nSummary:")
    print(f"  Total papers processed: {total_papers}")
    print(f"  Total graph nodes: {total_nodes}")
    print(f"  Total graph edges: {total_edges}")
    
    print(f"\nOutputs saved to:")
    print(f"  - Processed entities: {PROCESSED_DIR}/")
    print(f"  - Knowledge graphs: {GRAPHS_DIR}/")
    print(f"  - Training report: {DATA_DIR / 'training_report.json'}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. View knowledge graphs:")
    print(f"   open {GRAPHS_DIR}/*.png")
    print("\n2. Run the semantic matcher demo:")
    print("   python demos/demo_matcher.py")
    print("\n3. Run the landscape agent demo (now faster with cached entities):")
    print("   python demos/demo_landscape.py")
    print("\n")


if __name__ == "__main__":
    main()

