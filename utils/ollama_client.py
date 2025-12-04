"""
Ollama Client for LLM Interactions

Handles all interactions with local Ollama models for entity extraction and reasoning.
"""

import ollama
from typing import List, Dict, Optional
from loguru import logger
import json


class OllamaClient:
    """Client for interacting with Ollama LLMs."""
    
    def __init__(self, model: str = "gemma3:12b", temperature: float = 0.1):
        """
        Initialize Ollama client.
        
        Args:
            model: Ollama model name (e.g., "gemma3:12b", "llama3.1:8b")
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.model = model
        self.temperature = temperature
        logger.info(f"Ollama client initialized with model: {model}")
    
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The user prompt
            system: Optional system message
        
        Returns:
            Generated text
        """
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature}
            )
            
            return response["message"]["content"]
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text (drugs, targets, pathways, outcomes).
        
        Args:
            text: Medical text (e.g., abstract)
        
        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        system = """You are a medical entity extraction system. Extract entities from biomedical text and return them as JSON.
        
Entity types:
- drugs: Drug names, compounds, therapies (e.g., "Pembrolizumab", "Temozolomide", "CAR-T therapy")
- targets: Molecular targets, proteins, receptors (e.g., "PD-1", "EGFR", "IDH1")
- pathways: Biological pathways, mechanisms (e.g., "immune checkpoint", "angiogenesis", "DNA repair")
- outcomes: Clinical outcomes, results (e.g., "increased survival", "tumor regression", "adverse effects")

Return ONLY valid JSON in this exact format:
{
  "drugs": ["drug1", "drug2"],
  "targets": ["target1", "target2"],
  "pathways": ["pathway1"],
  "outcomes": ["outcome1", "outcome2"]
}

If no entities found for a category, use an empty list."""
        
        prompt = f"""Extract medical entities from this text:

{text}

Return JSON only, no additional text."""
        
        try:
            response = self.generate(prompt, system)
            
            # Try to parse JSON from response
            # Sometimes LLMs wrap JSON in markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code block
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.replace("```json", "").replace("```", "").strip()
            
            entities = json.loads(response)
            
            # Validate structure
            required_keys = ["drugs", "targets", "pathways", "outcomes"]
            for key in required_keys:
                if key not in entities:
                    entities[key] = []
            
            logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")
            return entities
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response was: {response}")
            return {"drugs": [], "targets": [], "pathways": [], "outcomes": []}
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"drugs": [], "targets": [], "pathways": [], "outcomes": []}
    
    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict]:
        """
        Extract relationships between entities.
        
        Args:
            text: Medical text
            entities: Previously extracted entities
        
        Returns:
            List of relationship triples (source, relation, target)
        """
        entity_str = "\n".join([f"- {k}: {', '.join(v)}" for k, v in entities.items() if v])
        
        system = """You are a biomedical relationship extraction system. Given entities and text, identify relationships between them.

Relationship types:
- targets: Drug/therapy targets a protein/pathway
- inhibits: Entity inhibits another entity
- activates: Entity activates another entity
- associated_with: Entity is associated with an outcome
- regulates: Entity regulates another entity

Return ONLY valid JSON array in this format:
[
  {"source": "entity1", "relation": "targets", "target": "entity2"},
  {"source": "entity2", "relation": "associated_with", "target": "outcome1"}
]

If no relationships found, return []."""
        
        prompt = f"""Text: {text}

Entities:
{entity_str}

Extract relationships between these entities. Return JSON only."""
        
        try:
            response = self.generate(prompt, system)
            
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.replace("```json", "").replace("```", "").strip()
            
            relationships = json.loads(response)
            
            if not isinstance(relationships, list):
                relationships = []
            
            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships
        
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []
    
    def compare_literature_to_trials(
        self,
        literature_summary: str,
        trial_summaries: List[str]
    ) -> str:
        """
        Compare literature findings to trial designs to identify discrepancies.
        
        Args:
            literature_summary: Summary of literature findings
            trial_summaries: List of trial summaries
        
        Returns:
            Analysis text highlighting discrepancies
        """
        trials_text = "\n\n".join([f"Trial {i+1}:\n{t}" for i, t in enumerate(trial_summaries)])
        
        system = """You are a clinical research analyst. Compare published literature to active clinical trials and identify potential discrepancies or misalignments.

Focus on:
1. Trials targeting mechanisms the literature suggests are ineffective
2. Trials using interventions with known adverse effects not mentioned
3. Trials contradicting recent scientific consensus
4. Promising approaches from literature not yet in trials

Be specific and cite evidence."""
        
        prompt = f"""Literature Summary:
{literature_summary}

Active Trials:
{trials_text}

Analyze and identify any discrepancies or noteworthy alignments/misalignments."""
        
        try:
            analysis = self.generate(prompt, system)
            return analysis
        except Exception as e:
            logger.error(f"Error comparing literature to trials: {e}")
            return "Error performing comparison analysis."


# Example usage
if __name__ == "__main__":
    client = OllamaClient(model="gemma3:12b")
    
    # Test entity extraction
    sample_text = """
    Pembrolizumab, an anti-PD-1 monoclonal antibody, showed increased overall survival 
    in patients with melanoma by targeting the PD-1/PD-L1 immune checkpoint pathway.
    """
    
    entities = client.extract_entities(sample_text)
    print("Entities:", json.dumps(entities, indent=2))
    
    relationships = client.extract_relationships(sample_text, entities)
    print("Relationships:", json.dumps(relationships, indent=2))

