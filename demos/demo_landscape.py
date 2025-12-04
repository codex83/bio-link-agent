"""
Demo: Therapeutic Landscape Agent

Demonstrates the Macro problem solution by analyzing 3 conditions:
1. Glioblastoma
2. Lung Cancer
3. Type 2 Diabetes

Shows how the agent compares literature to active trials to identify discrepancies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.landscape_agent import TherapeuticLandscapeAgent
from loguru import logger
import json


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert results to JSON-serializable format
    json_results = {
        "condition": results.get("condition", ""),
        "intervention": results.get("intervention", ""),
        "articles_analyzed": results.get("articles_analyzed", 0),
        "trials_analyzed": results.get("trials_analyzed", 0),
        "graph_summary": results.get("graph_summary", {}),
        "literature_summary": results.get("literature_summary", ""),
        "comparison_analysis": results.get("comparison_analysis", ""),
        "top_trials": [
            {
                "nct_id": t["nct_id"],
                "title": t["brief_title"],
                "status": t["status"]
            }
            for t in results.get("trials", [])[:5]
        ],
        "top_articles": [
            {
                "pmid": a["pmid"],
                "title": a["title"],
                "year": a["year"]
            }
            for a in results.get("articles", [])[:5]
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {filepath}")


def print_results(results: dict):
    """Print formatted results."""
    print("\n" + "="*80)
    print(f"THERAPEUTIC LANDSCAPE ANALYSIS: {results['condition'].upper()}")
    if results.get('intervention'):
        print(f"Intervention Focus: {results['intervention']}")
    print("="*80)
    
    if "error" in results:
        print(f"\nError: {results['error']}")
        return
    
    print(f"\nArticles Analyzed: {results['articles_analyzed']}")
    print(f"Active Trials Found: {results['trials_analyzed']}")
    
    graph = results['graph_summary']
    print(f"\nKnowledge Graph:")
    print(f"  - Total Entities: {graph['num_nodes']}")
    print(f"  - Relationships: {graph['num_edges']}")
    print(f"  - Drugs: {graph['num_drugs']}")
    print(f"  - Targets: {graph['num_targets']}")
    print(f"  - Pathways: {graph['num_pathways']}")
    print(f"  - Outcomes: {graph['num_outcomes']}")
    
    print("\n" + "-"*80)
    print("LITERATURE SUMMARY:")
    print("-"*80)
    print(results['literature_summary'])
    
    print("\n" + "-"*80)
    print("COMPARISON ANALYSIS:")
    print("-"*80)
    print(results['comparison_analysis'])
    
    print("\n" + "-"*80)
    print("TOP TRIALS:")
    print("-"*80)
    for i, trial in enumerate(results.get('trials', [])[:3], 1):
        print(f"\n{i}. {trial['brief_title']}")
        print(f"   NCT: {trial['nct_id']}")
        print(f"   Status: {trial['status']}")
        interventions = ", ".join([interv['name'] for interv in trial['interventions'][:3]])
        print(f"   Interventions: {interventions}")
    
    print("\n" + "="*80)


def main():
    """Run landscape agent demos for all 3 conditions."""
    
    # Initialize agent (will use cached data if available)
    agent = TherapeuticLandscapeAgent(
        ollama_model="gemma3:12b",
        max_papers=60,  # Fallback limit if cache not available
        max_trials=300,  # Fallback limit if cache not available
        use_cache=True  # Prioritize cached data
    )
    
    # Define conditions to analyze
    conditions = [
        {"condition": "glioblastoma", "intervention": "immunotherapy"},
        {"condition": "lung cancer", "intervention": "targeted therapy"},
        {"condition": "type 2 diabetes", "intervention": ""},
    ]
    
    print("\n" + "="*80)
    print("THERAPEUTIC LANDSCAPE AGENT - DEMO")
    print("="*80)
    print("\nThis demo will analyze 3 medical conditions:")
    print("1. Glioblastoma + Immunotherapy")
    print("2. Lung Cancer + Targeted Therapy")
    print("3. Type 2 Diabetes")
    print("\nFor each condition, the agent will:")
    print("  - Load cached papers (60 papers per condition)")
    print("  - Load pre-processed entities and knowledge graphs (if available)")
    print("  - Load cached trials (300 trials per condition)")
    print("  - Compare literature findings to trial designs")
    print("  - Identify potential discrepancies")
    print("\nNote: This uses cached data from data_pipeline.py and train.py")
    print("If cache is missing, it will fetch from APIs (slower).")
    print("\nThis may take 5-10 minutes...\n")
    
    input("Press Enter to start the demo...")
    
    # Run analysis for each condition
    all_results = []
    
    for i, config in enumerate(conditions, 1):
        print(f"\n{'='*80}")
        print(f"DEMO {i}/3: {config['condition'].upper()}")
        print(f"{'='*80}\n")
        
        try:
            results = agent.analyze_condition(
                condition=config['condition'],
                intervention=config['intervention']
            )
            
            # Print results
            print_results(results)
            
            # Save results
            filename = f"landscape_{config['condition'].replace(' ', '_')}.json"
            save_results(results, filename)
            
            # Save graph visualization
            graph_filename = f"demo_outputs/graph_{config['condition'].replace(' ', '_')}.png"
            agent.visualize_graph(graph_filename)
            print(f"Graph visualization saved to: {graph_filename}")
            
            all_results.append(results)
        
        except Exception as e:
            logger.error(f"Error analyzing {config['condition']}: {e}")
            print(f"\nError: {e}")
            continue
        
        if i < len(conditions):
            print("\n" + "-"*80)
            input("Press Enter to continue to next condition...")
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print(f"\nAnalyzed {len(all_results)} conditions successfully")
    print(f"Results saved to: demo_outputs/")
    print("\nKey Insights:")
    for result in all_results:
        condition = result.get('condition', 'Unknown')
        articles = result.get('articles_analyzed', 0)
        trials = result.get('trials_analyzed', 0)
        print(f"  - {condition.capitalize()}: {articles} papers, {trials} trials")
    print("\n")


if __name__ == "__main__":
    main()

