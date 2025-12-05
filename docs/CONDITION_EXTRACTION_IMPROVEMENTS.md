# Improving Condition Extraction in Eligibility Parser

## Current State

The current condition extraction uses simple regex patterns to find conditions mentioned in eligibility criteria:

```python
condition_keywords = [
    r'histologically\s+confirmed\s+([^,\.]+)',
    r'diagnosed\s+with\s+([^,\.]+)',
    r'history\s+of\s+([^,\.]+)',
    r'active\s+([^,\.]+?)\s+(?:disease|infection|cancer)',
]
```

**Limitations:**
- Extracts too much text (e.g., "non-small cell lung cancer (NSCLC)\n    3" instead of just "non-small cell lung cancer")
- Doesn't normalize condition names (e.g., "NSCLC" vs "non-small cell lung cancer")
- Doesn't handle abbreviations and synonyms
- Doesn't use medical ontologies (UMLS, SNOMED-CT)

## Proposed Improvements

### 1. **Named Entity Recognition (NER) with Biomedical Models**

Use specialized biomedical NER models to extract conditions more accurately:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class BiomedicalNER:
    def __init__(self):
        # Use a biomedical NER model
        self.model_name = "dmis-lab/biobert-base-cased-v1.1"
        # Or use: "allenai/scibert_scivocab_uncased"
        # Or use: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Load a fine-tuned NER model (would need to train or find pre-trained)
        # self.model = AutoModelForTokenClassification.from_pretrained(...)
    
    def extract_conditions(self, text: str) -> List[str]:
        # Tokenize and run NER
        # Extract entities labeled as "DISEASE", "CONDITION", etc.
        pass
```

**Benefits:**
- More accurate extraction
- Handles context better
- Can identify condition boundaries more precisely

### 2. **Medical Ontology Integration (UMLS/SNOMED-CT)**

Use medical ontologies to normalize and expand condition names:

```python
from umls_api import UmlsApi

class OntologyBasedExtractor:
    def __init__(self):
        # Initialize UMLS API (requires API key)
        self.umls = UmlsApi(api_key="your_key")
        # Or use local SNOMED-CT database
        
    def normalize_condition(self, condition_text: str) -> Dict:
        """
        Normalize condition using UMLS/SNOMED-CT.
        
        Returns:
            {
                "preferred_name": "Non-Small Cell Lung Carcinoma",
                "cui": "C0007131",  # Concept Unique Identifier
                "synonyms": ["NSCLC", "non-small cell lung cancer", ...],
                "semantic_types": ["Neoplastic Process"]
            }
        """
        # Search UMLS for concept
        results = self.umls.search(condition_text)
        
        # Get preferred name and synonyms
        concept = results[0] if results else None
        if concept:
            return {
                "preferred_name": concept.preferred_name,
                "cui": concept.cui,
                "synonyms": concept.synonyms,
                "semantic_types": concept.semantic_types
            }
        return None
    
    def is_subtype_of(self, condition1: str, condition2: str) -> bool:
        """
        Check if condition1 is a subtype of condition2 using ontology hierarchy.
        """
        # Use UMLS/SNOMED-CT hierarchical relationships
        # e.g., "NSCLC" is a subtype of "Lung Cancer"
        pass
```

**Benefits:**
- Normalizes variations ("NSCLC" = "non-small cell lung cancer")
- Handles synonyms automatically
- Enables hierarchical reasoning (subtype checking)

### 3. **Improved Regex with Context Awareness**

Enhance current regex patterns to be more precise:

```python
def _extract_conditions_improved(self, text: str) -> List[str]:
    """Improved condition extraction with better boundaries."""
    conditions = []
    
    # Pattern 1: "Histologically confirmed [CONDITION]"
    pattern1 = r'histologically\s+confirmed\s+([A-Z][^,\.;\(\)]+?)(?:\s*\([^\)]+\))?(?:\s*according|\s*per|\s*as|\s*with|[,\.;]|$)'
    
    # Pattern 2: "Diagnosed with [CONDITION]"
    pattern2 = r'diagnosed\s+with\s+([A-Z][^,\.;\(\)]+?)(?:\s*\([^\)]+\))?(?:\s*[,\.;]|$)'
    
    # Pattern 3: "History of [CONDITION]"
    pattern3 = r'history\s+of\s+([A-Z][^,\.;\(\)]+?)(?:\s*[,\.;]|$)'
    
    # Pattern 4: Specific cancer types
    cancer_patterns = [
        r'([A-Z][a-z]+\s+(?:cell\s+)?(?:lung|breast|colon|prostate|pancreatic)\s+cancer)',
        r'([A-Z]+(?:\s+[A-Z]+)?\s+(?:cancer|carcinoma|sarcoma|lymphoma|leukemia))',
    ]
    
    for pattern in [pattern1, pattern2, pattern3] + cancer_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            condition = match.group(1).strip()
            # Clean up common artifacts
            condition = re.sub(r'\s+', ' ', condition)  # Normalize whitespace
            condition = re.sub(r'^\d+\.\s*', '', condition)  # Remove numbering
            condition = re.sub(r'\s*\([^\)]*\)$', '', condition)  # Remove trailing parentheticals
            
            if 5 < len(condition) < 100:  # Reasonable length
                conditions.append(condition)
    
    return list(set(conditions))  # Remove duplicates
```

**Benefits:**
- Better boundary detection
- Handles parentheticals and abbreviations
- More precise extraction

### 4. **Hybrid Approach: Regex + LLM Post-Processing**

Use regex for initial extraction, then LLM to clean and normalize:

```python
from utils.ollama_client import OllamaClient

def extract_conditions_hybrid(self, text: str) -> List[str]:
    """Extract conditions using regex + LLM normalization."""
    
    # Step 1: Initial extraction with regex
    raw_conditions = self._extract_conditions(text)
    
    # Step 2: Use LLM to clean and normalize
    prompt = f"""
    Extract and normalize medical conditions from the following list.
    For each condition, provide:
    1. The standardized name
    2. Common abbreviations
    3. Whether it's a subtype of a broader condition
    
    Conditions: {', '.join(raw_conditions[:10])}
    
    Return as JSON:
    {{
        "conditions": [
            {{
                "standard_name": "...",
                "abbreviations": ["..."],
                "parent_condition": "..."
            }}
        ]
    }}
    """
    
    ollama = OllamaClient()
    response = ollama.generate(prompt)
    
    # Parse LLM response and extract normalized conditions
    # ...
    
    return normalized_conditions
```

**Benefits:**
- Leverages LLM understanding of medical terminology
- Can handle complex cases regex misses
- Provides normalization and relationship extraction

### 5. **Dictionary-Based Lookup**

Maintain a curated dictionary of common conditions:

```python
CONDITION_DICTIONARY = {
    "NSCLC": {
        "full_name": "Non-Small Cell Lung Cancer",
        "synonyms": ["non-small cell lung cancer", "non small cell lung cancer"],
        "parent": "Lung Cancer",
        "cui": "C0007131"
    },
    "SCLC": {
        "full_name": "Small Cell Lung Cancer",
        "synonyms": ["small cell lung cancer"],
        "parent": "Lung Cancer",
        "cui": "C0006826"
    },
    # ... more conditions
}

def normalize_with_dictionary(self, condition_text: str) -> Optional[str]:
    """Normalize condition using dictionary lookup."""
    condition_lower = condition_text.lower().strip()
    
    # Direct match
    for key, value in CONDITION_DICTIONARY.items():
        if condition_lower == key.lower() or condition_lower in [s.lower() for s in value["synonyms"]]:
            return value["full_name"]
    
    # Partial match
    for key, value in CONDITION_DICTIONARY.items():
        if any(syn.lower() in condition_lower for syn in value["synonyms"]):
            return value["full_name"]
    
    return None
```

**Benefits:**
- Fast lookup
- Handles common abbreviations
- No external API needed

## Recommended Implementation Strategy

1. **Short-term (Quick Win):**
   - Improve regex patterns with better boundaries (Option 3)
   - Add dictionary-based normalization for common conditions (Option 5)

2. **Medium-term:**
   - Integrate biomedical NER model (Option 1)
   - Add LLM post-processing for complex cases (Option 4)

3. **Long-term:**
   - Full UMLS/SNOMED-CT integration (Option 2)
   - Build comprehensive condition ontology

## Example Implementation

Here's a combined approach:

```python
class ImprovedConditionExtractor:
    def __init__(self):
        self.parser = EligibilityParser()
        self.condition_dict = CONDITION_DICTIONARY
        # Optionally: self.ner_model = BiomedicalNER()
        # Optionally: self.umls = UmlsApi()
    
    def extract_conditions(self, text: str) -> List[Dict]:
        """
        Extract and normalize conditions.
        
        Returns:
            List of condition dictionaries with normalized names
        """
        # Step 1: Initial extraction
        raw_conditions = self._extract_conditions_improved(text)
        
        # Step 2: Normalize using dictionary
        normalized = []
        for raw in raw_conditions:
            normalized_name = self.normalize_with_dictionary(raw)
            if normalized_name:
                normalized.append({
                    "raw": raw,
                    "normalized": normalized_name,
                    "source": "dictionary"
                })
            else:
                # Keep raw if no match found
                normalized.append({
                    "raw": raw,
                    "normalized": raw,
                    "source": "raw"
                })
        
        return normalized
```

This approach provides immediate improvements while leaving room for more sophisticated methods later.

