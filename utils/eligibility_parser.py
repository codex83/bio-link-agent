"""
Eligibility Criteria Parser

Parses structured eligibility criteria from clinical trial protocols.
Extracts inclusion/exclusion criteria and validates patient eligibility.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available. NER will not work.")


@dataclass
class LabValueConstraint:
    """Represents a laboratory value constraint."""
    parameter: str  # e.g., "hemoglobin", "creatinine"
    operator: str  # ">=", "<=", ">", "<", "=="
    value: float
    unit: Optional[str] = None  # e.g., "g/dL", "mg/dL"
    condition: Optional[str] = None  # e.g., "if liver metastases"


@dataclass
class EligibilityCriteria:
    """Structured representation of eligibility criteria."""
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    sex: Optional[str] = None  # "MALE", "FEMALE", "ALL"
    conditions: List[str] = field(default_factory=list)
    excluded_conditions: List[str] = field(default_factory=list)
    lab_constraints: List[LabValueConstraint] = field(default_factory=list)
    medication_exclusions: List[str] = field(default_factory=list)
    prior_treatment_requirements: List[str] = field(default_factory=list)
    performance_status: Optional[str] = None  # e.g., "ECOG 0-1", "Karnofsky >= 60%"
    pregnancy_requirement: Optional[str] = None  # e.g., "not pregnant", "effective contraception"


class BiomedicalNER:
    """
    Biomedical Named Entity Recognition for extracting medical conditions.
    Uses BC5CDR fine-tuned BioBERT model for disease/condition entity recognition.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize biomedical NER model.
        
        Args:
            model_name: HuggingFace model name for biomedical NER.
                       Default: "alvaroalon2/biobert_chemical_disease_ner" (BC5CDR)
                       Alternative: "alvaroalon2/biobert_disease_ner" (disease only)
        """
        if not TRANSFORMERS_AVAILABLE:
            self.ner_pipeline = None
            logger.warning("Transformers not available. NER disabled.")
            return
        
        # Use BC5CDR/NIH fine-tuned model for disease/condition NER
        # Try models in order of preference:
        # 1. raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed (BC5CDR trained)
        # 2. alvaroalon2/biobert_diseases_ner (disease NER)
        # 3. Ishan0612/biobert-ner-disease-ncbi (NCBI disease)
        
        model_options = [
            model_name,  # User-specified model first
            "raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed",  # BC5CDR trained
            "alvaroalon2/biobert_diseases_ner",  # Disease NER
            "Ishan0612/biobert-ner-disease-ncbi",  # NCBI disease
        ]
        
        # Remove None values
        model_options = [m for m in model_options if m is not None]
        
        self.ner_pipeline = None
        self.model_name = None
        
        for model_option in model_options:
            try:
                logger.info(f"Loading biomedical NER model: {model_option}")
                logger.info("This may take a few minutes on first run (downloading model)...")
                
                # Create NER pipeline with the fine-tuned model
                # aggregation_strategy="simple" groups subword tokens into complete words
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=model_option,
                    aggregation_strategy="simple",
                    device=-1  # Use CPU (set to 0 for GPU if available)
                )
                
                self.model_name = model_option
                logger.info(f"Successfully loaded NER model: {self.model_name}")
                break  # Success, exit loop
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_option}: {e}")
                if model_option == model_options[-1]:
                    # Last model failed
                    logger.error("All NER models failed to load. NER will be disabled.")
                    self.ner_pipeline = None
                    self.model_name = None
                continue  # Try next model
    
    def extract_conditions(self, text: str) -> List[str]:
        """
        Extract medical conditions from text using BC5CDR NER model.
        
        Uses fine-tuned BioBERT model to identify disease/condition entities.
        No regex patterns - pure NER-based extraction.
        
        Args:
            text: Input text containing eligibility criteria
        
        Returns:
            List of extracted condition names
        """
        if not TRANSFORMERS_AVAILABLE or self.ner_pipeline is None:
            logger.warning("NER pipeline not available")
            return []
        
        conditions = []
        
        try:
            # Run NER on the text
            # The model will identify entities and label them
            entities = self.ner_pipeline(text)
            
            # Filter for disease/condition entities
            # Different models use different label formats:
            # - "DISEASE", "B-DISEASE", "I-DISEASE" (BIO format)
            # - "Disease", "disease" (various cases)
            # - "O" (outside entity - skip)
            
            for entity in entities:
                entity_label = entity.get("entity_group", "")
                entity_word = entity.get("word", "").strip()
                
                # Normalize label for comparison
                label_upper = entity_label.upper() if entity_label else ""
                
                # Extract disease/condition entities
                # Accept various label formats used by different models
                is_disease = (
                    "DISEASE" in label_upper or
                    label_upper.startswith("B-") and "DISEASE" in label_upper or
                    label_upper.startswith("I-") and "DISEASE" in label_upper
                ) and label_upper != "O"  # Exclude "outside" label
                
                if is_disease and entity_word and len(entity_word) > 2:
                    # Clean up the entity
                    # Remove extra whitespace
                    entity_word = ' '.join(entity_word.split())
                    
                    # Basic validation - must have some alphabetic characters
                    if 3 < len(entity_word) < 150 and any(c.isalpha() for c in entity_word):
                        conditions.append(entity_word)
            
            # Also check for entities labeled as "CHEMICAL" that might be conditions
            # (some conditions might be mislabeled, but we'll be conservative)
            # For now, we'll focus on DISEASE labels
            
            # Remove duplicates while preserving order
            seen = set()
            unique_conditions = []
            for cond in conditions:
                # Normalize for comparison (lowercase, remove extra spaces)
                cond_normalized = ' '.join(cond.lower().split())
                if cond_normalized not in seen:
                    seen.add(cond_normalized)
                    unique_conditions.append(cond)
            
            logger.debug(f"NER extracted {len(unique_conditions)} unique conditions")
            return unique_conditions
            
        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            # Return empty list on error (fail gracefully)
            return []


class EligibilityParser:
    """
    Parser for clinical trial eligibility criteria.
    
    Extracts structured information from free-text eligibility criteria.
    Uses NER models for accurate condition extraction.
    """
    
    def __init__(self, use_ner: bool = True):
        """
        Initialize the eligibility parser.
        
        Args:
            use_ner: Whether to use NER models for condition extraction (default: True)
        """
        self.use_ner = use_ner
        if use_ner:
            self.ner_model = BiomedicalNER()
        else:
            self.ner_model = None
        # Common lab value patterns (handles escaped characters like \<, \>, \=)
        # Also handles unicode symbols like ≥, ≤, and various formats
        self.lab_patterns = {
            # Hemoglobin - handles Hb, hemoglobin, escaped and unicode operators
            r'hemoglobin\s*(?:\(Hb\))?\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:x\s*)?(?:ULN|upper limit of normal|g/dL|g/L)?': 'hemoglobin',
            r'(?:Hb|hemoglobin)\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:x\s*)?(?:ULN|upper limit of normal|g/dL|g/L)?': 'hemoglobin',
            
            # Platelet count - handles various formats
            r'platelet\s*(?:count|PLT)?\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:x?\s*10\^?[69]/L|K/mcL|/mcL|x\s*ULN)?': 'platelet',
            r'platelets?\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)': 'platelet',
            
            # White blood cell count
            r'white\s*blood\s*cell\s*(?:count|WBC)?\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)': 'wbc',
            r'WBC\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)': 'wbc',
            
            # Neutrophil/ANC
            r'(?:absolute\s*)?neutrophil\s*(?:count|ANC)?\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)': 'neutrophil',
            r'ANC\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)': 'neutrophil',
            
            # Creatinine
            r'creatinine\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:x\s*)?(?:ULN|upper limit of normal|mg/dL)?': 'creatinine',
            r'creatinine\s*clearance\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:mL/min|mg/dL)?': 'creatinine_clearance',
            
            # Bilirubin
            r'(?:total\s*)?bilirubin\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:x\s*)?(?:ULN|upper limit of normal)?': 'bilirubin',
            
            # Liver enzymes (AST, ALT, SGOT, SGPT)
            r'(?:AST|ALT|SGOT|SGPT)\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:x\s*)?(?:ULN|upper limit of normal)?': 'liver_enzyme',
            
            # Albumin
            r'albumin\s*(?:≥|>=|≤|<=|>|<|=|\\[<>]|\\=)\s*(\d+(?:\.\d+)?)\s*(?:g/dL|g/L)?': 'albumin',
        }
        
        # Age patterns
        self.age_patterns = [
            r'age\s*(?:of|≥|>=|>|≤|<=|<)?\s*(\d+)\s*(?:years?|yrs?)?',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*age|old)?',
            r'≥\s*(\d+)\s*(?:years?|yrs?)',
            r'>=?\s*(\d+)\s*(?:years?|yrs?)',
        ]
        
        # Performance status patterns
        self.performance_patterns = [
            r'ECOG\s*(?:score|performance\s*status)?\s*(?:of|≤|<=|≥|>=|==)?\s*(\d+(?:\s*-\s*\d+)?)',
            r'Karnofsky\s*(?:performance\s*status|score)?\s*(?:≥|>=|≤|<=|>|<)?\s*(\d+)',
        ]
        
        logger.info("Eligibility parser initialized")
    
    def parse(self, eligibility_text: str, min_age_str: str = "", max_age_str: str = "", sex_str: str = "") -> EligibilityCriteria:
        """
        Parse eligibility criteria from text.
        
        Args:
            eligibility_text: Raw eligibility criteria text
            min_age_str: Minimum age string (e.g., "18 Years")
            max_age_str: Maximum age string (e.g., "120 Years")
            sex_str: Sex requirement (e.g., "ALL", "MALE", "FEMALE")
        
        Returns:
            EligibilityCriteria object with structured data
        """
        criteria = EligibilityCriteria()
        
        # Parse age from separate fields first
        criteria.min_age = self._parse_age(min_age_str)
        criteria.max_age = self._parse_age(max_age_str)
        criteria.sex = sex_str.upper() if sex_str else None
        
        if not eligibility_text:
            return criteria
        
        # Split into inclusion and exclusion sections
        inclusion_text, exclusion_text = self._split_criteria(eligibility_text)
        
        # Parse inclusion criteria
        if inclusion_text:
            criteria.inclusion_criteria = self._extract_criteria_list(inclusion_text)
            criteria.conditions.extend(self._extract_conditions(inclusion_text))
            criteria.lab_constraints.extend(self._extract_lab_values(inclusion_text))
            criteria.performance_status = self._extract_performance_status(inclusion_text)
            criteria.prior_treatment_requirements.extend(self._extract_prior_treatment(inclusion_text))
            criteria.pregnancy_requirement = self._extract_pregnancy_requirement(inclusion_text)
        
        # Parse exclusion criteria
        if exclusion_text:
            criteria.exclusion_criteria = self._extract_criteria_list(exclusion_text)
            criteria.excluded_conditions.extend(self._extract_conditions(exclusion_text))
            criteria.medication_exclusions.extend(self._extract_medications(exclusion_text))
            criteria.lab_constraints.extend(self._extract_lab_values(exclusion_text))
        
        # Extract age from text if not already set
        if not criteria.min_age:
            criteria.min_age = self._extract_age_from_text(inclusion_text or eligibility_text)
        
        return criteria
    
    def _split_criteria(self, text: str) -> Tuple[str, str]:
        """Split eligibility text into inclusion and exclusion sections."""
        text = text.strip()
        
        # Common patterns for splitting
        inclusion_patterns = [
            r'INCLUSION\s*CRITERIA[:\s]*(.*?)(?=EXCLUSION\s*CRITERIA|$)',
            r'Inclusion\s*Criteria[:\s]*(.*?)(?=Exclusion\s*Criteria|$)',
            r'Eligibility[:\s]*(.*?)(?=Exclusion|$)',
        ]
        
        exclusion_patterns = [
            r'EXCLUSION\s*CRITERIA[:\s]*(.*)',
            r'Exclusion\s*Criteria[:\s]*(.*)',
        ]
        
        inclusion_text = ""
        exclusion_text = ""
        
        # Try to find inclusion section
        for pattern in inclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                inclusion_text = match.group(1).strip()
                break
        
        # Try to find exclusion section
        for pattern in exclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                exclusion_text = match.group(1).strip()
                break
        
        # If no clear split, assume all text is inclusion
        if not inclusion_text and not exclusion_text:
            inclusion_text = text
        
        return inclusion_text, exclusion_text
    
    def _extract_criteria_list(self, text: str) -> List[str]:
        """Extract individual criteria items from text."""
        criteria = []
        
        # Split by numbered items or bullets
        # Pattern: "1.", "2.", "*", "-", etc.
        items = re.split(r'\n\s*(?:\d+\.|\*|\-)\s+', text)
        
        for item in items:
            item = item.strip()
            if item and len(item) > 10:  # Filter out very short items
                # Clean up the item
                item = re.sub(r'^\d+\.\s*', '', item)
                item = re.sub(r'^\*\s*', '', item)
                item = re.sub(r'^-\s*', '', item)
                criteria.append(item.strip())
        
        return criteria
    
    def _extract_conditions(self, text: str) -> List[str]:
        """
        Extract medical conditions mentioned in criteria using NER.
        
        Args:
            text: Eligibility criteria text
        
        Returns:
            List of extracted condition names
        """
        if self.use_ner and self.ner_model:
            # Use NER model for extraction
            conditions = self.ner_model.extract_conditions(text)
            logger.debug(f"NER extracted {len(conditions)} conditions")
            return conditions
        else:
            # Fallback: return empty list if NER not available
            logger.warning("NER not available, returning empty condition list")
            return []
    
    def _extract_lab_values(self, text: str) -> List[LabValueConstraint]:
        """Extract laboratory value constraints."""
        constraints = []
        
        # First, normalize escaped characters and unicode symbols
        # Replace \< with <, \> with >, \= with =, etc.
        normalized_text = text.replace('\\<', '<').replace('\\>', '>').replace('\\=', '=')
        normalized_text = normalized_text.replace('≥', '>=').replace('≤', '<=')
        
        for pattern, param_name in self.lab_patterns.items():
            # Normalize pattern to handle escaped characters
            normalized_pattern = pattern.replace('\\[<>]', '[<>]').replace('\\=', '=')
            normalized_pattern = normalized_pattern.replace('≥', '>=').replace('≤', '<=')
            
            matches = re.finditer(normalized_pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract operator from the match
                    full_match = match.group(0)
                    value_str = match.group(1).strip()
                    
                    # Determine operator from the match
                    if '>=' in full_match or '≥' in full_match:
                        operator = '>='
                    elif '<=' in full_match or '≤' in full_match:
                        operator = '<='
                    elif '>' in full_match:
                        operator = '>'
                    elif '<' in full_match:
                        operator = '<'
                    else:
                        operator = '>='  # Default
                    
                    # Extract unit if present in the match
                    unit = None
                    if 'g/dL' in full_match:
                        unit = 'g/dL'
                    elif 'mg/dL' in full_match:
                        unit = 'mg/dL'
                    elif 'ULN' in full_match or 'upper limit of normal' in full_match:
                        unit = 'x ULN'
                    elif '/L' in full_match or '/mcL' in full_match:
                        unit = '/L'
                    
                    value = float(value_str)
                    constraints.append(LabValueConstraint(
                        parameter=param_name,
                        operator=operator,
                        value=value,
                        unit=unit
                    ))
                except (ValueError, IndexError, AttributeError):
                    continue
        
        return constraints
    
    def _extract_age_from_text(self, text: str) -> Optional[int]:
        """Extract minimum age from text."""
        for pattern in self.age_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    age = int(match.group(1))
                    if 0 <= age <= 120:  # Reasonable age range
                        return age
                except (ValueError, IndexError):
                    continue
        return None
    
    def _parse_age(self, age_str: str) -> Optional[int]:
        """Parse age from string like '18 Years'."""
        if not age_str:
            return None
        
        match = re.search(r'(\d+)', age_str)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_performance_status(self, text: str) -> Optional[str]:
        """Extract performance status requirement."""
        for pattern in self.performance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medication exclusions or requirements."""
        medications = []
        
        # Patterns for medication mentions
        patterns = [
            r'prior\s+(?:treatment\s+with|therapy\s+with|use\s+of)\s+([^,\.]+)',
            r'receiving\s+([^,\.]+?)\s+(?:therapy|treatment|medication)',
            r'concurrent\s+(?:use\s+of|treatment\s+with)\s+([^,\.]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                med = match.group(1).strip()
                if len(med) > 2 and len(med) < 50:
                    medications.append(med)
        
        return list(set(medications))
    
    def _extract_prior_treatment(self, text: str) -> List[str]:
        """Extract prior treatment requirements."""
        treatments = []
        
        patterns = [
            r'prior\s+(?:treatment|therapy)\s+(?:with|of)\s+([^,\.]+)',
            r'failed\s+(?:prior|previous)\s+([^,\.]+)',
            r'refractory\s+to\s+([^,\.]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                treatment = match.group(1).strip()
                if len(treatment) > 3:
                    treatments.append(treatment)
        
        return list(set(treatments))
    
    def _extract_pregnancy_requirement(self, text: str) -> Optional[str]:
        """Extract pregnancy/contraception requirements."""
        if re.search(r'pregnant|pregnancy|contraception|childbearing', text, re.IGNORECASE):
            # Extract the relevant sentence
            sentences = re.split(r'[\.\n]', text)
            for sentence in sentences:
                if re.search(r'pregnant|pregnancy|contraception|childbearing', sentence, re.IGNORECASE):
                    return sentence.strip()[:200]  # Limit length
        return None
    
    def validate_patient(
        self,
        criteria: EligibilityCriteria,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
        patient_conditions: Optional[List[str]] = None,
        patient_labs: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate patient eligibility against criteria.
        
        Args:
            criteria: Parsed eligibility criteria
            patient_age: Patient age
            patient_sex: Patient sex ("MALE", "FEMALE")
            patient_conditions: List of patient conditions
            patient_labs: Dictionary of lab values {parameter: value}
        
        Returns:
            Tuple of (is_eligible, list_of_reasons)
        """
        reasons = []
        is_eligible = True
        
        # Check age
        if patient_age is not None:
            if criteria.min_age and patient_age < criteria.min_age:
                is_eligible = False
                reasons.append(f"Age {patient_age} below minimum {criteria.min_age}")
            if criteria.max_age and patient_age > criteria.max_age:
                is_eligible = False
                reasons.append(f"Age {patient_age} above maximum {criteria.max_age}")
        
        # Check sex
        if patient_sex and criteria.sex:
            if criteria.sex not in ["ALL", patient_sex.upper()]:
                is_eligible = False
                reasons.append(f"Sex {patient_sex} does not match requirement {criteria.sex}")
        
        # Check excluded conditions
        if patient_conditions and criteria.excluded_conditions:
            for condition in patient_conditions:
                for excluded in criteria.excluded_conditions:
                    if condition.lower() in excluded.lower() or excluded.lower() in condition.lower():
                        is_eligible = False
                        reasons.append(f"Condition '{condition}' is excluded: {excluded}")
        
        # Check lab values
        if patient_labs and criteria.lab_constraints:
            for constraint in criteria.lab_constraints:
                if constraint.parameter in patient_labs:
                    value = patient_labs[constraint.parameter]
                    if constraint.operator == '>=' and value < constraint.value:
                        is_eligible = False
                        reasons.append(f"{constraint.parameter} {value} below required {constraint.operator} {constraint.value}")
                    elif constraint.operator == '<=' and value > constraint.value:
                        is_eligible = False
                        reasons.append(f"{constraint.parameter} {value} above required {constraint.operator} {constraint.value}")
        
        return is_eligible, reasons


# Example usage
if __name__ == "__main__":
    parser = EligibilityParser()
    
    sample_criteria = """
    INCLUSION CRITERIA:
    1. Age >= 18 years
    2. Histologically confirmed lung cancer
    3. ECOG performance status 0-1
    4. Hemoglobin >= 9 g/dL
    5. Platelet count >= 100,000/mcL
    
    EXCLUSION CRITERIA:
    1. Pregnant or breastfeeding
    2. Active infection
    3. Prior treatment with immunotherapy
    """
    
    criteria = parser.parse(sample_criteria, min_age_str="18 Years", sex_str="ALL")
    print(f"Min Age: {criteria.min_age}")
    print(f"Lab Constraints: {len(criteria.lab_constraints)}")
    print(f"Inclusion Criteria: {len(criteria.inclusion_criteria)}")
    print(f"Exclusion Criteria: {len(criteria.exclusion_criteria)}")

