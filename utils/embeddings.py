"""
Embedding and Vector Search Utilities

Handles embedding generation and vector database operations for semantic search.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from loguru import logger
import os
import torch


def get_best_device() -> str:
    """
    Detect the best available device for PyTorch embeddings.
    
    Note: For small models like all-MiniLM-L6-v2, CPU is actually faster
    than GPU due to data transfer overhead. GPU is only beneficial for
    very large models or batch sizes > 1000.
    """
    # CPU is faster for small embedding models (tested: CPU 0.2s vs MPS 3.6s)
    # GPU overhead isn't worth it for 22M parameter models
    logger.info("Using CPU for embeddings (optimal for small models)")
    return "cpu"
    
    # Uncomment below to force GPU (only useful for very large batches)
    # if torch.backends.mps.is_available():
    #     return "mps"
    # elif torch.cuda.is_available():
    #     return "cuda"
    # return "cpu"


class EmbeddingManager:
    """Manages embeddings and vector search operations."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./data/indexed_trials",
        device: Optional[str] = None
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Sentence transformer model name
            persist_directory: Directory to persist vector store
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # Auto-detect best device if not specified
        self.device = device or get_best_device()
        
        # Initialize sentence transformer with GPU support
        logger.info(f"Loading embedding model: {model_name} on device: {self.device}")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        # Initialize ChromaDB
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        logger.info(f"Embedding manager initialized (device: {self.device})")
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            List of embedding vectors
        """
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def create_collection(self, collection_name: str, overwrite: bool = False):
        """
        Create or get a collection in the vector database.
        
        Args:
            collection_name: Name of the collection
            overwrite: If True, delete existing collection and create new one
        
        Returns:
            ChromaDB collection
        """
        if overwrite:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass
        
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Collection '{collection_name}' ready")
        return collection
    
    def index_trials(
        self,
        trials: List[Dict],
        collection_name: str = "clinical_trials",
        overwrite: bool = False
    ):
        """
        Index clinical trials into vector database.
        
        Args:
            trials: List of trial dictionaries
            collection_name: Name of the collection
            overwrite: Whether to overwrite existing collection
        """
        collection = self.create_collection(collection_name, overwrite)
        
        # Prepare documents for indexing
        documents = []
        metadatas = []
        ids = []
        
        for trial in trials:
            nct_id = trial.get("nct_id", "")
            if not nct_id:
                continue
            
            # Create searchable text from trial information
            title = trial.get("brief_title", "")
            conditions = ", ".join(trial.get("conditions", []))
            eligibility = trial.get("eligibility_criteria", "")
            summary = trial.get("brief_summary", "")
            
            # Combine into searchable document
            document = f"""
            Title: {title}
            Conditions: {conditions}
            Eligibility Criteria: {eligibility}
            Summary: {summary}
            """.strip()
            
            documents.append(document)
            
            # Store metadata
            metadata = {
                "nct_id": nct_id,
                "title": title,
                "status": trial.get("status", ""),
                "conditions": conditions,
                "min_age": trial.get("min_age", ""),
                "max_age": trial.get("max_age", ""),
                "sex": trial.get("sex", ""),
            }
            metadatas.append(metadata)
            ids.append(nct_id)
        
        # Encode documents
        logger.info(f"Encoding {len(documents)} trial documents")
        embeddings = self.encode_texts(documents)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        logger.info(f"Indexed {len(documents)} trials into '{collection_name}'")
    
    def search_trials(
        self,
        query: str,
        collection_name: str = "clinical_trials",
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for trials using semantic similarity.
        
        Args:
            query: Search query (e.g., patient description)
            collection_name: Name of the collection to search
            top_k: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of matching trials with scores
        """
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Collection '{collection_name}' not found: {e}")
            return []
        
        # Encode query
        query_embedding = self.encode_texts([query])[0]
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Format results
        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "nct_id": results["ids"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i],
            })
        
        logger.info(f"Found {len(matches)} matching trials")
        return matches
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            return {
                "name": collection_name,
                "count": count,
                "model": self.model_name,
            }
        except Exception as e:
            logger.error(f"Error getting stats for '{collection_name}': {e}")
            return {"name": collection_name, "count": 0, "error": str(e)}


class OntologyChecker:
    """Simple rule-based ontology checker for trial eligibility."""
    
    def __init__(self):
        """Initialize ontology checker with basic medical hierarchies."""
        # Simple hierarchies (in production, use UMLS/SNOMED)
        self.cancer_hierarchy = {
            "lung cancer": ["non-small cell lung cancer", "small cell lung cancer", "nsclc", "sclc"],
            "solid tumor": ["lung cancer", "breast cancer", "colon cancer", "glioblastoma"],
            "brain cancer": ["glioblastoma", "astrocytoma", "medulloblastoma"],
        }
        
        self.condition_synonyms = {
            "shortness of breath": ["dyspnea", "breathing difficulty"],
            "high blood sugar": ["hyperglycemia", "diabetes"],
            "tired": ["fatigue", "exhaustion"],
        }
        
        logger.info("Ontology checker initialized")
    
    def is_subtype_of(self, specific: str, general: str) -> bool:
        """
        Check if specific condition is a subtype of general condition.
        
        Args:
            specific: Specific condition (e.g., "lung cancer")
            general: General condition (e.g., "solid tumor")
        
        Returns:
            True if specific is a subtype of general
        """
        specific = specific.lower().strip()
        general = general.lower().strip()
        
        if specific == general:
            return True
        
        # Check hierarchy
        if general in self.cancer_hierarchy:
            return specific in self.cancer_hierarchy[general]
        
        return False
    
    def get_synonyms(self, term: str) -> List[str]:
        """
        Get synonyms for a medical term.
        
        Args:
            term: Medical term
        
        Returns:
            List of synonyms (including the original term)
        """
        term = term.lower().strip()
        synonyms = [term]
        
        if term in self.condition_synonyms:
            synonyms.extend(self.condition_synonyms[term])
        
        return synonyms
    
    def check_exclusion(self, patient_condition: str, exclusion_criteria: str) -> bool:
        """
        Check if patient condition matches exclusion criteria.
        
        Args:
            patient_condition: Patient's condition
            exclusion_criteria: Trial exclusion criteria text
        
        Returns:
            True if patient should be excluded
        """
        exclusion_lower = exclusion_criteria.lower()
        
        # Check direct mentions
        if patient_condition.lower() in exclusion_lower:
            return True
        
        # Check if patient condition is a subtype of excluded condition
        for general_condition in ["solid tumor", "cancer", "brain cancer"]:
            if general_condition in exclusion_lower:
                if self.is_subtype_of(patient_condition, general_condition):
                    return True
        
        return False


# Example usage
if __name__ == "__main__":
    # Test embedding manager
    manager = EmbeddingManager()
    
    # Test encoding
    texts = ["Patient with shortness of breath and cough", "Diabetes Type 2 management"]
    embeddings = manager.encode_texts(texts)
    print(f"Encoded {len(embeddings)} texts, embedding dim: {len(embeddings[0])}")
    
    # Test ontology checker
    checker = OntologyChecker()
    print(f"Is lung cancer a solid tumor? {checker.is_subtype_of('lung cancer', 'solid tumor')}")
    print(f"Synonyms for 'tired': {checker.get_synonyms('tired')}")

