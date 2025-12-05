"""
Test eligibility parser with real trial data.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eligibility_parser import EligibilityParser

def test_with_real_trials():
    """Test parser with real trial data from cache."""
    parser = EligibilityParser()
    
    # Load a real trial
    trials_file = Path("data/trials/lung_cancer_trials.json")
    
    if not trials_file.exists():
        print(f"ERROR: Trial file not found: {trials_file}")
        print("Please run data pipeline first to cache trials.")
        return
    
    with open(trials_file, 'r') as f:
        trials = json.load(f)
    
    print("="*80)
    print("ELIGIBILITY PARSER TEST WITH REAL TRIAL DATA")
    print("="*80)
    
    # Test with first 3 trials
    for i, trial in enumerate(trials[:3], 1):
        print(f"\n{'='*80}")
        print(f"TRIAL {i}: {trial.get('brief_title', 'N/A')[:70]}")
        print(f"{'='*80}")
        print(f"NCT ID: {trial.get('nct_id', 'N/A')}")
        print(f"Status: {trial.get('status', 'N/A')}")
        print(f"Conditions: {', '.join(trial.get('conditions', []))}")
        print(f"Min Age: {trial.get('min_age', 'N/A')}")
        print(f"Max Age: {trial.get('max_age', 'N/A')}")
        print(f"Sex: {trial.get('sex', 'N/A')}")
        
        # Parse eligibility criteria
        eligibility_text = trial.get('eligibility_criteria', '')
        if not eligibility_text:
            print("\n⚠ No eligibility criteria found in trial data")
            continue
        
        print(f"\n--- Parsing Eligibility Criteria (first 500 chars) ---")
        print(eligibility_text[:500] + "..." if len(eligibility_text) > 500 else eligibility_text)
        
        criteria = parser.parse(
            eligibility_text=eligibility_text,
            min_age_str=trial.get('min_age', ''),
            max_age_str=trial.get('max_age', ''),
            sex_str=trial.get('sex', '')
        )
        
        print(f"\n--- Parsed Results ---")
        print(f"✓ Min Age: {criteria.min_age}")
        print(f"✓ Max Age: {criteria.max_age}")
        print(f"✓ Sex: {criteria.sex}")
        print(f"✓ Inclusion Criteria Items: {len(criteria.inclusion_criteria)}")
        print(f"✓ Exclusion Criteria Items: {len(criteria.exclusion_criteria)}")
        print(f"✓ Lab Value Constraints: {len(criteria.lab_constraints)}")
        print(f"✓ Conditions Extracted: {len(criteria.conditions)}")
        print(f"✓ Excluded Conditions: {len(criteria.excluded_conditions)}")
        print(f"✓ Medication Exclusions: {len(criteria.medication_exclusions)}")
        print(f"✓ Performance Status: {criteria.performance_status}")
        
        if criteria.inclusion_criteria:
            print(f"\n--- Sample Inclusion Criteria (first 2) ---")
            for j, inc in enumerate(criteria.inclusion_criteria[:2], 1):
                print(f"  {j}. {inc[:100]}...")
        
        if criteria.exclusion_criteria:
            print(f"\n--- Sample Exclusion Criteria (first 2) ---")
            for j, exc in enumerate(criteria.exclusion_criteria[:2], 1):
                print(f"  {j}. {exc[:100]}...")
        
        if criteria.lab_constraints:
            print(f"\n--- Lab Value Constraints ---")
            for lab in criteria.lab_constraints[:5]:  # Show first 5
                print(f"  • {lab.parameter}: {lab.operator} {lab.value} {lab.unit or ''}")
        
        # Test patient validation
        print(f"\n--- Patient Validation Tests ---")
        
        # Test Case 1: Eligible patient
        test_patient_1 = {
            "age": 55,
            "sex": "MALE",
            "conditions": ["non-small cell lung cancer"],
            "labs": {"hemoglobin": 12.0, "platelet": 150000}
        }
        is_eligible, reasons = parser.validate_patient(
            criteria=criteria,
            patient_age=test_patient_1["age"],
            patient_sex=test_patient_1["sex"],
            patient_conditions=test_patient_1["conditions"],
            patient_labs=test_patient_1["labs"]
        )
        print(f"  Test 1 - 55yo male with NSCLC: {'✓ ELIGIBLE' if is_eligible else '✗ INELIGIBLE'}")
        if reasons:
            print(f"    Reasons: {', '.join(reasons)}")
        
        # Test Case 2: Too young
        test_patient_2 = {
            "age": 16,
            "sex": "MALE",
            "conditions": ["non-small cell lung cancer"]
        }
        is_eligible, reasons = parser.validate_patient(
            criteria=criteria,
            patient_age=test_patient_2["age"],
            patient_sex=test_patient_2["sex"],
            patient_conditions=test_patient_2["conditions"]
        )
        print(f"  Test 2 - 16yo male: {'✓ ELIGIBLE' if is_eligible else '✗ INELIGIBLE'}")
        if reasons:
            print(f"    Reasons: {', '.join(reasons)}")
        
        # Test Case 3: Wrong sex (if trial has sex restriction)
        if criteria.sex and criteria.sex != "ALL":
            test_patient_3 = {
                "age": 45,
                "sex": "FEMALE" if criteria.sex == "MALE" else "MALE",
                "conditions": ["non-small cell lung cancer"]
            }
            is_eligible, reasons = parser.validate_patient(
                criteria=criteria,
                patient_age=test_patient_3["age"],
                patient_sex=test_patient_3["sex"],
                patient_conditions=test_patient_3["conditions"]
            )
            print(f"  Test 3 - Wrong sex ({test_patient_3['sex']}): {'✓ ELIGIBLE' if is_eligible else '✗ INELIGIBLE'}")
            if reasons:
                print(f"    Reasons: {', '.join(reasons)}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_with_real_trials()

