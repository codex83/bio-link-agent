"""
Test script for eligibility parser.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eligibility_parser import EligibilityParser

def test_basic_parsing():
    """Test basic eligibility criteria parsing."""
    parser = EligibilityParser()
    
    sample_criteria = """
    INCLUSION CRITERIA:
    1. Age >= 18 years
    2. Histologically confirmed non-small cell lung cancer (NSCLC)
    3. ECOG performance status 0-1
    4. Hemoglobin >= 9 g/dL
    5. Platelet count >= 100,000/mcL
    6. Adequate renal function (creatinine <= 1.5 mg/dL)
    
    EXCLUSION CRITERIA:
    1. Pregnant or breastfeeding women
    2. Active infection requiring treatment
    3. Prior treatment with immunotherapy within 6 months
    4. History of autoimmune disease
    """
    
    criteria = parser.parse(
        eligibility_text=sample_criteria,
        min_age_str="18 Years",
        max_age_str="",
        sex_str="ALL"
    )
    
    print("=== Parsing Test Results ===")
    print(f"Min Age: {criteria.min_age}")
    print(f"Max Age: {criteria.max_age}")
    print(f"Sex: {criteria.sex}")
    print(f"Inclusion Criteria Count: {len(criteria.inclusion_criteria)}")
    print(f"Exclusion Criteria Count: {len(criteria.exclusion_criteria)}")
    print(f"Lab Constraints: {len(criteria.lab_constraints)}")
    print(f"Performance Status: {criteria.performance_status}")
    print(f"Conditions: {criteria.conditions[:3]}")
    
    # Test validation
    print("\n=== Validation Test ===")
    
    # Eligible patient
    is_eligible, reasons = parser.validate_patient(
        criteria=criteria,
        patient_age=45,
        patient_sex="MALE",
        patient_conditions=["non-small cell lung cancer"],
        patient_labs={"hemoglobin": 12.0, "platelet": 150000, "creatinine": 1.2}
    )
    print(f"Eligible patient (45yo male, NSCLC): {is_eligible}")
    if reasons:
        print(f"  Reasons: {reasons}")
    
    # Ineligible patient (too young)
    is_eligible, reasons = parser.validate_patient(
        criteria=criteria,
        patient_age=16,
        patient_sex="MALE",
        patient_conditions=["non-small cell lung cancer"]
    )
    print(f"\nIneligible patient (16yo): {is_eligible}")
    if reasons:
        print(f"  Reasons: {reasons}")
    
    # Ineligible patient (low hemoglobin)
    is_eligible, reasons = parser.validate_patient(
        criteria=criteria,
        patient_age=45,
        patient_sex="MALE",
        patient_conditions=["non-small cell lung cancer"],
        patient_labs={"hemoglobin": 7.0}
    )
    print(f"\nIneligible patient (low Hb): {is_eligible}")
    if reasons:
        print(f"  Reasons: {reasons}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_basic_parsing()

