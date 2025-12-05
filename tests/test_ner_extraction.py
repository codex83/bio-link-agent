"""
Test NER-based condition extraction with real eligibility criteria.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eligibility_parser import BiomedicalNER, EligibilityParser

def test_ner_extraction():
    """Test NER model on real eligibility criteria."""
    
    print("="*80)
    print("TESTING BC5CDR NER MODEL FOR CONDITION EXTRACTION")
    print("="*80)
    
    # Initialize NER model
    print("\n1. Initializing NER model...")
    ner = BiomedicalNER()
    
    if ner.ner_pipeline is None:
        print("ERROR: NER pipeline not available")
        return
    
    print(f"✓ Model loaded: {ner.model_name}")
    
    # Test with sample eligibility criteria
    print("\n2. Testing with sample eligibility criteria...")
    
    sample_text = """
    INCLUSION CRITERIA:
    1. Histologically confirmed non-small cell lung cancer (NSCLC), adenocarcinoma subtype
    2. Patients with glioblastoma multiforme (GBM), grade IV
    3. Type 2 diabetes mellitus with poorly controlled blood sugar
    4. Metastatic breast cancer that has progressed after prior therapy
    
    EXCLUSION CRITERIA:
    1. Active infection requiring treatment
    2. History of autoimmune disease
    3. Pregnant or breastfeeding women
    """
    
    print("Sample text:")
    print(sample_text[:200] + "...")
    
    conditions = ner.extract_conditions(sample_text)
    print(f"\n✓ Extracted {len(conditions)} conditions:")
    for i, cond in enumerate(conditions, 1):
        print(f"  {i}. {cond}")
    
    # Test with real trial data
    print("\n3. Testing with real trial eligibility criteria...")
    
    trials_file = Path("data/trials/lung_cancer_trials.json")
    if trials_file.exists():
        with open(trials_file, 'r') as f:
            trials = json.load(f)
        
        # Test with first trial
        trial = trials[0]
        eligibility = trial.get('eligibility_criteria', '')
        
        if eligibility:
            print(f"\nTrial: {trial.get('brief_title', 'N/A')[:60]}...")
            print(f"Eligibility criteria (first 300 chars): {eligibility[:300]}...")
            
            conditions = ner.extract_conditions(eligibility)
            print(f"\n✓ Extracted {len(conditions)} conditions:")
            for i, cond in enumerate(conditions[:10], 1):  # Show first 10
                print(f"  {i}. {cond}")
            if len(conditions) > 10:
                print(f"  ... and {len(conditions) - 10} more")
    else:
        print("  Trial data not found, skipping real data test")
    
    # Test with EligibilityParser integration
    print("\n4. Testing EligibilityParser with NER...")
    parser = EligibilityParser(use_ner=True)
    
    if trials_file.exists():
        trial = trials[0]
        criteria = parser.parse(
            eligibility_text=trial.get('eligibility_criteria', ''),
            min_age_str=trial.get('min_age', ''),
            max_age_str=trial.get('max_age', ''),
            sex_str=trial.get('sex', '')
        )
        
        print(f"✓ Parsed criteria:")
        print(f"  - Conditions extracted: {len(criteria.conditions)}")
        if criteria.conditions:
            print(f"  - Sample conditions: {', '.join(criteria.conditions[:5])}")
        print(f"  - Excluded conditions: {len(criteria.excluded_conditions)}")
        if criteria.excluded_conditions:
            print(f"  - Sample excluded: {', '.join(criteria.excluded_conditions[:3])}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_ner_extraction()

