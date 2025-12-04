"""
Therapeutic Landscape Agent (Macro Problem)

This agent solves the "macro gap" by:
1. Querying PubMed for recent literature
2. Extracting entities and building a knowledge graph
3. Querying ClinicalTrials.gov for active trials
4. Comparing literature findings to trial designs
5. Identifying discrepancies and misalignments
"""

import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.pubmed_server import PubMedServer
from mcp_servers.clinicaltrials_server import ClinicalTrialsServer
from utils.ollama_client import OllamaClient
from utils.graph_builder import KnowledgeGraphBuilder
from typing import Dict, List
from loguru import logger
from tqdm import tqdm


class TherapeuticLandscapeAgent:
    """
    Agent for analyzing therapeutic landscape by comparing literature to trials.
    """
    
    def __init__(
        self,
        ollama_model: str = "gemma3:12b",
        max_papers: int = 25,
        max_trials: int = 20,
        use_cache: bool = True
    ):
        """
        Initialize the landscape agent.
        
        Args:
            ollama_model: Ollama model to use
            max_papers: Maximum number of papers to analyze (fallback if cache not available)
            max_trials: Maximum number of trials to analyze (fallback if cache not available)
            use_cache: Whether to prioritize cached data over API calls
        """
        self.pubmed = PubMedServer()
        self.clinicaltrials = ClinicalTrialsServer()
        self.llm = OllamaClient(model=ollama_model)
        self.graph_builder = KnowledgeGraphBuilder()
        
        self.max_papers = max_papers
        self.max_trials = max_trials
        self.use_cache = use_cache
        
        # Data directories
        self.data_dir = Path("./data")
        self.papers_dir = self.data_dir / "papers"
        self.trials_dir = self.data_dir / "trials"
        self.processed_dir = self.data_dir / "processed"
        self.graphs_dir = self.data_dir / "graphs"
        
        logger.info("Therapeutic Landscape Agent initialized")
    
    def analyze_condition(self, condition: str, intervention: str = "") -> Dict:
        """
        Analyze a medical condition by comparing literature to active trials.
        
        Args:
            condition: Medical condition (e.g., "glioblastoma")
            intervention: Optional intervention filter (e.g., "immunotherapy")
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting analysis for: {condition} {intervention}")
        
        # Normalize condition name for file paths
        condition_normalized = condition.replace(" ", "_").lower()
        
        # Step 1: Load papers (from cache or API)
        search_query = f"{condition} {intervention}".strip()
        articles = []
        
        if self.use_cache:
            papers_file = self.papers_dir / f"{condition_normalized}_papers.json"
            if papers_file.exists():
                logger.info(f"Loading cached papers from {papers_file}")
                with open(papers_file, 'r') as f:
                    articles = json.load(f)
                logger.info(f"Loaded {len(articles)} cached papers")
            else:
                logger.warning(f"Cached papers not found for {condition}. Fetching from API...")
        
        if not articles:
            logger.info(f"Querying PubMed: {search_query}")
            articles = self.pubmed.search_and_fetch(search_query, max_results=self.max_papers)
            if not articles:
                logger.warning("No articles found")
                return {"error": "No articles found for this query"}
            logger.info(f"Retrieved {len(articles)} articles from API")
        
        # Step 2: Load or extract entities and relationships
        logger.info("Loading/extracting entities from literature...")
        articles_with_entities = []
        
        # Try to load pre-processed entities
        processed_file = self.processed_dir / f"{condition_normalized}_entities.json"
        processed_papers = None
        
        if self.use_cache and processed_file.exists():
            logger.info(f"Loading pre-processed entities from {processed_file}")
            with open(processed_file, 'r') as f:
                processed_papers = json.load(f)
            logger.info(f"Loaded {len(processed_papers)} pre-processed papers")
        
        if processed_papers:
            # Match processed papers to articles by PMID
            article_dict = {a.get("pmid", ""): a for a in articles}
            for processed in processed_papers:
                pmid = processed.get("pmid", "")
                if pmid in article_dict:
                    articles_with_entities.append({
                        "article": article_dict[pmid],
                        "entities": processed.get("entities", {}),
                        "relationships": processed.get("relationships", [])
                    })
        else:
            # Extract entities on the fly
            logger.info("Extracting entities from articles (this may take a while)...")
            for article in tqdm(articles, desc="Processing articles"):
                abstract = article.get("abstract", "")
                if not abstract or len(abstract) < 50:
                    continue
                
                try:
                    entities = self.llm.extract_entities(abstract)
                    relationships = self.llm.extract_relationships(abstract, entities)
                    articles_with_entities.append({
                        "article": article,
                        "entities": entities,
                        "relationships": relationships
                    })
                except Exception as e:
                    logger.warning(f"Error processing article {article.get('pmid', 'unknown')}: {e}")
                    continue
        
        if not articles_with_entities:
            logger.warning("No articles with entities found")
            return {"error": "No articles with entities found"}
        
        # Step 3: Load or build knowledge graph
        logger.info("Loading/building knowledge graph...")
        graph_file = self.graphs_dir / f"{condition_normalized}_graph.pkl"
        
        if self.use_cache and graph_file.exists():
            logger.info(f"Loading pre-built graph from {graph_file}")
            if self.graph_builder.load_from_pickle(str(graph_file)):
                graph_summary = self.graph_builder.get_summary()
                logger.info(f"Graph loaded: {graph_summary}")
            else:
                logger.warning("Failed to load graph, building from scratch...")
                self.graph_builder = KnowledgeGraphBuilder()
                self.graph_builder.build_from_articles(articles_with_entities)
                graph_summary = self.graph_builder.get_summary()
                logger.info(f"Graph built: {graph_summary}")
        else:
            logger.info("Building knowledge graph from articles...")
            self.graph_builder.build_from_articles(articles_with_entities)
            graph_summary = self.graph_builder.get_summary()
            logger.info(f"Graph built: {graph_summary}")
        
        # Step 4: Load trials (from cache or API)
        trials = []
        
        if self.use_cache:
            trials_file = self.trials_dir / f"{condition_normalized}_trials.json"
            if trials_file.exists():
                logger.info(f"Loading cached trials from {trials_file}")
                with open(trials_file, 'r') as f:
                    all_trials = json.load(f)
                # Filter to RECRUITING status for analysis
                trials = [t for t in all_trials if t.get("status", "").upper() == "RECRUITING"]
                logger.info(f"Loaded {len(trials)} recruiting trials from cache (out of {len(all_trials)} total)")
            else:
                logger.warning(f"Cached trials not found for {condition}. Fetching from API...")
        
        if not trials:
            logger.info(f"Querying ClinicalTrials.gov: {search_query}")
            trials = self.clinicaltrials.search_and_fetch(
                search_query,
                max_results=self.max_trials,
                status="RECRUITING"
            )
            if not trials:
                logger.warning("No active trials found")
                return {
                    "error": "No active trials found",
                    "articles_analyzed": len(articles),
                    "graph_summary": graph_summary
                }
            logger.info(f"Retrieved {len(trials)} active trials from API")
        
        # Step 5: Generate literature summary
        logger.info("Generating literature summary...")
        literature_summary = self._summarize_literature(articles_with_entities, graph_summary)
        
        # Step 6: Generate trial summaries
        logger.info("Generating trial summaries...")
        trial_summaries = []
        for trial in trials:
            interventions = ", ".join([i["name"] for i in trial["interventions"]])
            summary = f"""
            {trial['brief_title']} (NCT: {trial['nct_id']})
            Status: {trial['status']}
            Interventions: {interventions}
            Summary: {trial['brief_summary'][:300]}...
            """.strip()
            trial_summaries.append(summary)
        
        # Step 7: Compare literature to trials
        logger.info("Comparing literature to trials...")
        comparison_analysis = self.llm.compare_literature_to_trials(
            literature_summary,
            trial_summaries
        )
        
        # Step 8: Compile results
        results = {
            "condition": condition,
            "intervention": intervention,
            "articles_analyzed": len(articles),
            "trials_analyzed": len(trials),
            "graph_summary": graph_summary,
            "literature_summary": literature_summary,
            "trials": trials[:5],  # Include top 5 trials
            "comparison_analysis": comparison_analysis,
            "articles": articles[:10],  # Include top 10 articles
        }
        
        logger.info("Analysis complete")
        return results
    
    def _summarize_literature(self, articles_with_entities: List[Dict], graph_summary: Dict) -> str:
        """
        Generate a summary of literature findings.
        
        Args:
            articles_with_entities: Articles with extracted entities
            graph_summary: Graph statistics
        
        Returns:
            Summary text
        """
        # Collect top entities
        all_drugs = []
        all_targets = []
        all_outcomes = []
        
        for item in articles_with_entities:
            entities = item["entities"]
            all_drugs.extend(entities.get("drugs", []))
            all_targets.extend(entities.get("targets", []))
            all_outcomes.extend(entities.get("outcomes", []))
        
        # Count occurrences
        from collections import Counter
        drug_counts = Counter(all_drugs)
        target_counts = Counter(all_targets)
        outcome_counts = Counter(all_outcomes)
        
        # Build summary
        top_drugs = ", ".join([d for d, _ in drug_counts.most_common(5)])
        top_targets = ", ".join([t for t, _ in target_counts.most_common(5)])
        top_outcomes = ", ".join([o for o, _ in outcome_counts.most_common(3)])
        
        summary = f"""
        Literature Analysis Summary:
        
        Articles Analyzed: {len(articles_with_entities)}
        Knowledge Graph: {graph_summary['num_nodes']} nodes, {graph_summary['num_edges']} relationships
        
        Most Mentioned Drugs/Therapies: {top_drugs if top_drugs else "None identified"}
        Most Mentioned Targets: {top_targets if top_targets else "None identified"}
        Common Outcomes: {top_outcomes if top_outcomes else "None identified"}
        
        Recent Findings:
        """
        
        # Add snippets from recent articles
        for i, item in enumerate(articles_with_entities[:3], 1):
            article = item["article"]
            summary += f"\n{i}. {article['title']} ({article['year']})"
            summary += f"\n   {article['abstract'][:200]}...\n"
        
        return summary.strip()
    
    def visualize_graph(self, output_path: str = "therapeutic_landscape.png"):
        """
        Visualize the knowledge graph.
        
        Args:
            output_path: Path to save the visualization
        """
        self.graph_builder.visualize(output_path)
        logger.info(f"Graph visualization saved to {output_path}")


# Example usage
if __name__ == "__main__":
    agent = TherapeuticLandscapeAgent(max_papers=10, max_trials=5)
    
    # Analyze glioblastoma immunotherapy
    results = agent.analyze_condition("glioblastoma", "immunotherapy")
    
    print("\n" + "="*80)
    print("THERAPEUTIC LANDSCAPE ANALYSIS")
    print("="*80)
    print(f"\nCondition: {results['condition']} {results['intervention']}")
    print(f"Articles Analyzed: {results['articles_analyzed']}")
    print(f"Trials Analyzed: {results['trials_analyzed']}")
    print(f"\nGraph: {results['graph_summary']}")
    print("\n" + "-"*80)
    print("LITERATURE SUMMARY:")
    print("-"*80)
    print(results['literature_summary'])
    print("\n" + "-"*80)
    print("COMPARISON ANALYSIS:")
    print("-"*80)
    print(results['comparison_analysis'])
    print("\n" + "="*80)
    
    # Visualize graph
    agent.visualize_graph()

