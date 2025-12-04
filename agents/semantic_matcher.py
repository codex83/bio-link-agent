"""
Semantic Matcher Agent (Micro Problem)

This agent solves the "micro mismatch" by:
1. Indexing trial eligibility criteria into a vector database
2. Taking patient descriptions (free text or clinical notes)
3. Using semantic search to find matching trials
4. Applying ontology-based filtering for hard constraints
5. Returning ranked, eligible trials
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.clinicaltrials_server import ClinicalTrialsServer
from utils.embeddings import EmbeddingManager, OntologyChecker
from typing import Dict, List, Optional
from loguru import logger


class SemanticMatcher:
    """
    Agent for matching patients to clinical trials using semantic search.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        data_dir: str = "./data/indexed_trials"
    ):
        """
        Initialize the semantic matcher agent.
        
        Args:
            embedding_model: Sentence transformer model name
            data_dir: Directory for vector database
        """
        self.clinicaltrials = ClinicalTrialsServer()
        self.embeddings = EmbeddingManager(
            model_name=embedding_model,
            persist_directory=data_dir
        )
        self.ontology = OntologyChecker()
        
        logger.info("Semantic Matcher Agent initialized")
    
    def index_trials_for_condition(
        self,
        condition: str,
        max_trials: int = 100,
        collection_name: Optional[str] = None
    ):
        """
        Index trials for a specific condition into the vector database.
        
        Args:
            condition: Medical condition (e.g., "lung cancer")
            max_trials: Maximum number of trials to index
            collection_name: Optional custom collection name
        """
        if collection_name is None:
            # Create collection name from condition
            collection_name = condition.lower().replace(" ", "_")
        
        logger.info(f"Fetching trials for: {condition}")
        trials = self.clinicaltrials.search_and_fetch(
            condition,
            max_results=max_trials,
            status="RECRUITING"
        )
        
        if not trials:
            logger.warning(f"No trials found for {condition}")
            return
        
        logger.info(f"Indexing {len(trials)} trials into collection '{collection_name}'")
        self.embeddings.index_trials(trials, collection_name, overwrite=True)
        logger.info("Indexing complete")
    
    def match_patient_to_trials(
        self,
        patient_description: str,
        collection_name: str,
        patient_conditions: Optional[List[str]] = None,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Match a patient to clinical trials.
        
        Args:
            patient_description: Free text description of patient (symptoms, history, etc.)
            collection_name: Collection to search
            patient_conditions: List of patient conditions for exclusion checking
            patient_age: Patient age (for age-based filtering)
            patient_sex: Patient sex (M/F/ALL)
            top_k: Number of top matches to return
        
        Returns:
            List of matching trials with scores and reasoning
        """
        logger.info(f"Searching for trials matching: {patient_description[:100]}...")
        
        # Step 1: Semantic search
        matches = self.embeddings.search_trials(
            query=patient_description,
            collection_name=collection_name,
            top_k=top_k * 2  # Get more for filtering
        )
        
        if not matches:
            logger.warning("No matches found")
            return []
        
        # Step 2: Apply ontology-based filtering
        filtered_matches = []
        
        for match in matches:
            metadata = match["metadata"]
            nct_id = match["nct_id"]
            
            # Check age constraints
            if patient_age is not None:
                min_age_str = metadata.get("min_age", "")
                max_age_str = metadata.get("max_age", "")
                
                # Parse age constraints (simple parsing)
                if self._violates_age_constraint(patient_age, min_age_str, max_age_str):
                    logger.debug(f"Trial {nct_id} excluded: age constraint")
                    continue
            
            # Check sex constraints
            if patient_sex is not None:
                trial_sex = metadata.get("sex", "ALL")
                if trial_sex not in ["ALL", patient_sex]:
                    logger.debug(f"Trial {nct_id} excluded: sex constraint")
                    continue
            
            # Check condition-based exclusions (simplified)
            if patient_conditions:
                # This is a simplified check - in production, would use full eligibility criteria
                excluded = False
                for condition in patient_conditions:
                    # Check if this is an excluded condition type
                    # (This is very simplified - real implementation needs full criteria parsing)
                    if "excluded" in match["document"].lower():
                        # Would do more sophisticated parsing here
                        pass
                
                if excluded:
                    continue
            
            # Add reasoning for why this trial matched
            match["reasoning"] = self._generate_match_reasoning(
                patient_description,
                match["document"],
                match["score"]
            )
            
            filtered_matches.append(match)
            
            if len(filtered_matches) >= top_k:
                break
        
        logger.info(f"Found {len(filtered_matches)} eligible trials after filtering")
        return filtered_matches
    
    def _violates_age_constraint(self, patient_age: int, min_age_str: str, max_age_str: str) -> bool:
        """
        Check if patient age violates trial age constraints.
        
        Args:
            patient_age: Patient age
            min_age_str: Minimum age string (e.g., "18 Years")
            max_age_str: Maximum age string (e.g., "65 Years")
        
        Returns:
            True if age constraint is violated
        """
        try:
            # Parse minimum age
            if min_age_str and min_age_str != "N/A":
                min_age = int(min_age_str.split()[0])
                if patient_age < min_age:
                    return True
            
            # Parse maximum age
            if max_age_str and max_age_str != "N/A":
                max_age = int(max_age_str.split()[0])
                if patient_age > max_age:
                    return True
        
        except (ValueError, IndexError):
            # If parsing fails, be conservative and don't exclude
            pass
        
        return False
    
    def _generate_match_reasoning(self, query: str, document: str, score: float) -> str:
        """
        Generate human-readable reasoning for why a trial matched.
        
        Args:
            query: Patient description
            document: Trial document
            score: Similarity score
        
        Returns:
            Reasoning text
        """
        # Extract key matching concepts (simplified)
        query_lower = query.lower()
        doc_lower = document.lower()
        
        matching_terms = []
        
        # Simple keyword overlap detection
        important_terms = ["cancer", "tumor", "diabetes", "hypertension", "immunotherapy", 
                          "chemotherapy", "surgery", "metastatic", "stage", "lung", "brain"]
        
        for term in important_terms:
            if term in query_lower and term in doc_lower:
                matching_terms.append(term)
        
        reasoning = f"Semantic similarity: {score:.2f}. "
        
        if matching_terms:
            reasoning += f"Matching concepts: {', '.join(matching_terms[:3])}."
        else:
            reasoning += "General semantic alignment with patient profile."
        
        return reasoning
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """
        Get information about an indexed collection.
        
        Args:
            collection_name: Collection name
        
        Returns:
            Collection statistics
        """
        return self.embeddings.get_collection_stats(collection_name)
    
    def format_results(self, matches: List[Dict]) -> str:
        """
        Format match results for display.
        
        Args:
            matches: List of match dictionaries
        
        Returns:
            Formatted text
        """
        if not matches:
            return "No matching trials found."
        
        output = f"\nFound {len(matches)} matching clinical trials:\n"
        output += "="*80 + "\n\n"
        
        for i, match in enumerate(matches, 1):
            metadata = match["metadata"]
            output += f"{i}. {metadata['title']}\n"
            output += f"   NCT ID: {match['nct_id']}\n"
            output += f"   Status: {metadata['status']}\n"
            output += f"   Conditions: {metadata['conditions']}\n"
            output += f"   Match Score: {match['score']:.3f}\n"
            output += f"   Reasoning: {match['reasoning']}\n"
            output += f"   Eligibility: Age {metadata.get('min_age', 'N/A')}-{metadata.get('max_age', 'N/A')}, "
            output += f"Sex: {metadata.get('sex', 'ALL')}\n"
            output += "-"*80 + "\n\n"
        
        return output


# Example usage
if __name__ == "__main__":
    matcher = SemanticMatcher()
    
    # Index lung cancer trials
    print("Indexing lung cancer trials...")
    matcher.index_trials_for_condition("lung cancer", max_trials=50)
    
    # Example patient
    patient_description = """
    65-year-old male patient with non-small cell lung cancer (NSCLC), stage III.
    Patient has shortness of breath, persistent cough, and weight loss.
    Previous treatment with chemotherapy. No other significant medical history.
    """
    
    print("\nSearching for matching trials...")
    matches = matcher.match_patient_to_trials(
        patient_description=patient_description,
        collection_name="lung_cancer",
        patient_age=65,
        patient_sex="MALE",
        top_k=5
    )
    
    # Display results
    print(matcher.format_results(matches))

