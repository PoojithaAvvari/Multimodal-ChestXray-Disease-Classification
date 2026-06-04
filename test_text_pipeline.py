#!/usr/bin/env python3
"""
Quick Test Script for Text Analysis Pipeline
Tests all 3 stages: Medical validation → CXR validation → Disease extraction
"""

import requests
import json
from pprint import pprint

API_URL = "http://127.0.0.1:8000/analyze"

# Test cases
TEST_CASES = {
    "1_medical_cxr_positive": {
        "text": """FINDINGS: Right lower lobe pneumonia with airspace consolidation. 
Heart size enlarged consistent with cardiomegaly. 
Small right pleural effusion. No pneumothorax identified.
IMPRESSION: Pneumonia with cardiomegaly and effusion.""",
        "expected": "success"
    },
    
    "2_medical_cxr_negated": {
        "text": """FINDINGS: No pneumonia. No consolidation. 
Heart size normal. No evidence of pleural effusion.
IMPRESSION: Chest X-ray unremarkable.""",
        "expected": "success (no findings)"
    },
    
    "3_not_medical": {
        "text": "This is a beautiful day. I went to the park and played football.",
        "expected": "rejected (model1_not_medical)"
    },
    
    "4_medical_not_cxr": {
        "text": """FINDINGS: Brain MRI shows a small lesion in the frontal lobe.
No hemorrhage. Mild brain atrophy.
IMPRESSION: Possible glioma.""",
        "expected": "rejected (model2_not_cxr)"
    },
    
    "5_short_text": {
        "text": "heart",
        "expected": "rejected (too short)"
    }
}

def test_report_analysis():
    print("\n" + "="*80)
    print("TESTING TEXT ANALYSIS PIPELINE")
    print("="*80)
    
    for test_name, test_data in TEST_CASES.items():
        print(f"\n\n{'='*80}")
        print(f"TEST: {test_name}")
        print(f"Expected: {test_data['expected']}")
        print(f"Input Text: {test_data['text'][:100]}...")
        print(f"{'='*80}")
        
        try:
            response = requests.post(
                API_URL,
                data={"report_text": test_data["text"]},
                timeout=60
            )
            
            print(f"\n✓ Response Status: {response.status_code}")
            result = response.json()
            
            # Pretty print the response
            print(f"\nResponse:")
            pprint(result, depth=2)
            
            # Check status
            if result.get("status") == "success":
                print("\n✅ SUCCESS - Diseases Found:")
                for disease in result.get("disease_findings", []):
                    print(f"   • {disease['disease']}: {disease['label']}")
                if result.get("ruled_out"):
                    print("\n📋 Ruled Out:")
                    for disease in result.get("ruled_out", []):
                        print(f"   • {disease['disease']}: {disease['label']}")
            else:
                print(f"\n❌ REJECTED - Reason: {result.get('reason', 'Unknown')}")
                print(f"   Step Failed: {result.get('step_failed', 'Unknown')}")
                if "model1" in result:
                    print(f"   Model 1 (Medical): {result['model1'].get('is_medical', False)} ({result['model1'].get('confidence', 0)*100:.1f}%)")
                if "model2" in result:
                    print(f"   Model 2 (CXR): {result['model2'].get('is_cxr', False)} ({result['model2'].get('confidence', 0)*100:.1f}%)")
                    
        except requests.exceptions.ConnectionError:
            print(f"\n❌ ERROR: Cannot connect to backend at {API_URL}")
            print("   Make sure backend is running: python main.py")
            return False
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            return False
    
    print(f"\n\n{'='*80}")
    print("✅ ALL TESTS COMPLETED")
    print(f"{'='*80}\n")
    return True

if __name__ == "__main__":
    import sys
    success = test_report_analysis()
    sys.exit(0 if success else 1)
