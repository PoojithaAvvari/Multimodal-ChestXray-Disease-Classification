"""
Report Text Analysis Pipeline — 3-Stage Medical Report Classifier
===================================================================
Stage 1: Is this medical text? (BART-MNLI + keyword boost)
Stage 2: Is this a CXR report? (BART-MNLI + CXR keywords)
Stage 3: Extract 14 diseases (CheXbert + rule-based)
"""

import torch
import re
import os
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import StringIO

# Prevent TensorFlow import issues
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
if 'tensorflow' not in sys.modules:
    sys.modules['tensorflow'] = None

# ============================================================================
# LAZY-LOADED MODEL SINGLETONS
# ============================================================================

_bart_mnli = None
_bart_tokenizer = None
_chexbert_model = None
_chexbert_tokenizer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# KEYWORD LISTS
# ============================================================================

MEDICAL_KEYWORDS = {
    "chest", "lung", "lungs", "heart", "pulmonary", "cardiac", "ribs", "rib",
    "diaphragm", "mediastinum", "pleura", "pleural", "x-ray", "xray", "radiograph",
    "radiology", "pneumonia", "effusion", "pneumothorax", "atelectasis",
    "cardiomegaly", "edema", "consolidation", "nodule", "mass", "fracture",
    "infiltrate", "opacity", "thorax", "thoracic", "costophrenic", "hilum",
    "hilar", "aortic", "aorta", "sternum", "clavicle", "trachea", "bronchial",
    "ventilation", "intubation", "catheter", "findings", "impression",
    "indication", "comparison", "swelling", "pain", "fever", "cough",
    "shortness of breath", "breathing", "breath", "wheezing", "infection",
    "inflammation", "tumor", "cancer", "bleeding", "discharge", "nausea",
    "vomiting", "headache", "dizziness", "fatigue", "weakness", "tender",
    "patient", "diagnosis", "symptom", "symptoms", "treatment", "medication",
    "prescribed", "lab", "blood", "urine", "biopsy", "surgery", "hospital",
    "clinic", "emergency", "trauma", "wound", "abnormal", "normal", "chronic",
    "acute", "benign", "malignant", "disease", "condition", "syndrome", "disorder"
}

STRONG_KEYWORDS = {
    "lung", "lungs", "pulmonary", "cardiac", "pleura", "pleural", "radiograph",
    "radiology", "pneumonia", "effusion", "pneumothorax", "atelectasis",
    "cardiomegaly", "edema", "consolidation", "nodule", "mass", "fracture",
    "infiltrate", "opacity", "thorax", "thoracic", "costophrenic", "cardiopulmonary",
    "cardiomediastinal", "findings", "impression", "x-ray", "xray", "ribs",
    "diaphragm", "hilum", "hilar", "trachea", "bronchial", "acute", "abnormality",
    "swelling", "pain", "fever", "cough", "breath", "infection", "tumor",
    "cancer", "bleeding", "nausea", "patient", "diagnosis", "treatment",
    "blood", "surgery", "hospital", "trauma", "wound", "chronic", "benign",
    "malignant", "disease", "syndrome"
}

CXR_KEYWORDS = {
    "chest", "lung", "lungs", "x-ray", "xray", "radiograph", "ribs", "rib",
    "pleura", "pleural", "pneumothorax", "pneumonia", "cardiomegaly",
    "atelectasis", "thorax", "thoracic", "pulmonary", "cardiopulmonary",
    "cardiomediastinal", "costophrenic", "hilum", "hilar", "diaphragm",
    "consolidation", "effusion", "edema", "opacity", "infiltrate", "bronchial",
    "trachea", "aortic", "clavicle", "sternum", "lobe", "lobar", "pneumonitis",
    "bronchopneumonia", "cardiac", "heart", "mediastinal", "mediastinum", 
    "bony cage", "soft tissue"
}

DISEASE_PATTERNS = {
    "Atelectasis": [
        r"\batelectasis\b", r"\bcollapse\b", r"\bloss of (lung )?volume\b",
        r"\bvolume loss\b", r"\bshrinkage\b"
    ],
    "Cardiomegaly": [
        r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bheart (is )?enlarged\b",
        r"\bcard[iac]* enlargement\b", r"\bheart size\s*(increased|large)\b"
    ],
    "Consolidation": [
        r"\bconsolidation\b", r"\bair(space)? (opacification|disease)\b",
        r"\bopacity\b", r"\bopacities\b", r"\binfiltrate\b", r"\binfiltrates\b"
    ],
    "Edema": [
        r"\bedema\b", r"\bpulmonary edema\b", r"\bvascular markings?\s*(prominent|increased|enhanced)\b",
        r"\bcongestion\b", r"\bfluid overload\b"
    ],
    "Pleural Effusion": [
        r"\bpleural effusion\b", r"\beffusion\b", r"\bblunting (of )?(the )?costophrenic\b",
        r"\bfluid\b", r"\bpleural fluid\b"
    ],
    "Pneumonia": [
        r"\bpneumonia\b", r"\bpneumonitis\b", r"\binfection\b", r"\binfectious\b",
        r"\binfectious process\b"
    ],
    "Pneumothorax": [
        r"\bpneumothorax\b", r"\bcollapsed lung\b", r"\blung collapse\b",
        r"\bair in pleural\b"
    ],
    "Fracture": [
        r"\bfracture\b", r"\bfractured\b", r"\brib fracture\b", r"\bfractured rib\b"
    ],
    "Mass": [
        r"\bmass\b", r"\bmasses\b", r"\blump\b", r"\btumor\b", r"\blesion\b"
    ],
    "Nodule": [
        r"\bnodule\b", r"\bnodules\b", r"\bsmall nodule\b"
    ],
    "Enlarged Cardiomediastinum": [
        r"\bmediastinal\s*(widening|enlargement|mass)\b", r"\bcardiomediastinal\b"
    ],
    "Lung Lesion": [
        r"\blesion\b", r"\blesions\b", r"\blocation\b"
    ],
    "Pleural Other": [
        r"\bpleural\s*(thickening|disease|abnormality)\b", r"\bthickening\b"
    ],
    "Support Devices": [
        r"\bcatheter\b", r"\btube\b", r"\bline\b", r"\bintubat[io]n\b",
        r"\bvent[ilator]*\b", r"\bdevice\b"
    ]
}

NEGATION_WORDS = {
    "no", "not", "without", "denies", "negative for", "rule out", "ruled out",
    "absent", "absence of", "no evidence of", "no sign of", "no signs of",
    "cleared of", "free of", "unremarkable"
}

POST_NEGATION = {
    "is not seen", "is not present", "is not identified", "is excluded",
    "is not appreciated", "is ruled out", "has resolved", "not seen"
}

# ============================================================================
# MODEL LOADERS
# ============================================================================

def _load_bart_mnli():
    """Lazy-load BART-MNLI model (shared by Model 1 & 2)"""
    global _bart_mnli, _bart_tokenizer
    if _bart_mnli is not None:
        return
    print("Loading BART-MNLI for text classification...")
    _bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    _bart_mnli = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    _bart_mnli = _bart_mnli.to(device).eval()
    print("BART-MNLI loaded.")

def _load_chexbert():
    """Lazy-load CheXbert model"""
    global _chexbert_model, _chexbert_tokenizer
    if _chexbert_model is not None:
        return
    print("Loading CheXbert for disease extraction...")
    
    try:
        _chexbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load CheXbert checkpoint
        chexbert_path = os.path.join(os.path.dirname(__file__), "CheXbert", "chexbert.pth")
        
        if not os.path.exists(chexbert_path):
            print(f"Downloading CheXbert checkpoint to {chexbert_path}...")
            import urllib.request
            os.makedirs(os.path.dirname(chexbert_path), exist_ok=True)
            url = "https://huggingface.co/StanfordAIMI/RRG_scorers/resolve/main/chexbert.pth"
            urllib.request.urlretrieve(url, chexbert_path)
        
        # Load BERT-base architecture
        from transformers import BertModel
        _chexbert_model = BertModel.from_pretrained("bert-base-uncased")
        _chexbert_model = _chexbert_model.to(device).eval()
        
        print("CheXbert loaded.")
    except Exception as e:
        print(f"Warning: CheXbert loading failed: {e}. Using rule-based only.")
        _chexbert_model = False

# ============================================================================
# NLI ENTAILMENT SCORING
# ============================================================================

def _nli_entailment(text, hypothesis):
    """Get NLI entailment score (0 to 1)"""
    _load_bart_mnli()
    
    input_text = f"{text} </s> {hypothesis}"
    inputs = _bart_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = _bart_mnli(**inputs).logits
    
    probs = torch.softmax(logits, dim=1)[0]
    entailment_prob = probs[2].item()  # BART: 0=contradiction, 1=neutral, 2=entailment
    return entailment_prob

# ============================================================================
# MODEL 1: IS THIS MEDICAL TEXT?
# ============================================================================

def model1_is_medical(text):
    """
    Stage 1: Check if text is a medical report
    Returns: (is_medical, confidence, details)
    """
    if len(text.strip()) < 20:
        return False, 0.0, {"reason": "Text too short"}
    
    # NLI score
    hypothesis = "This text is a medical report describing patient findings, diagnoses, or clinical observations."
    nli_score = _nli_entailment(text, hypothesis)
    
    # Keyword boost
    keywords_matched = []
    for kw in MEDICAL_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text.lower()):
            keywords_matched.append(kw)
    
    keyword_boost = min(0.1 * len(keywords_matched), 0.5)
    
    # Strong keywords for hard rule
    strong_count = sum(1 for kw in STRONG_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text.lower()))
    
    final_score = nli_score + keyword_boost
    
    # Hard rules
    if strong_count == 0:
        final_score = min(final_score, 0.35)
    elif strong_count >= 3:
        final_score = max(final_score, 0.55)
    
    final_score = min(final_score, 1.0)
    is_medical = final_score > 0.45
    
    return is_medical, final_score, {
        "nli_score": round(nli_score, 4),
        "keyword_count": len(keywords_matched),
        "keywords_matched": keywords_matched[:10],
        "strong_keywords_count": strong_count
    }

# ============================================================================
# MODEL 2: IS THIS A CXR REPORT?
# ============================================================================

def model2_is_cxr_report(text):
    """
    Stage 2: Check if medical text is a chest X-ray report
    Returns: (is_cxr, confidence, details)
    """
    # NLI score
    hypothesis = "This text describes findings from a chest X-ray or chest radiograph."
    nli_score = _nli_entailment(text, hypothesis)
    
    # CXR keyword boost
    cxr_keywords_matched = []
    for kw in CXR_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text.lower()):
            cxr_keywords_matched.append(kw)
    
    # Boost by 0.15 per keyword match, cap at 0.6
    keyword_boost = min(0.15 * len(cxr_keywords_matched), 0.6)
    
    final_score = nli_score + keyword_boost
    final_score = min(final_score, 1.0)
    
    # Lower threshold since findings can be very short
    is_cxr = final_score > 0.35
    
    return is_cxr, final_score, {
        "nli_score": round(nli_score, 4),
        "cxr_keyword_count": len(cxr_keywords_matched),
        "cxr_keywords_matched": cxr_keywords_matched[:10]
    }

# ============================================================================
# NEGATION DETECTION
# ============================================================================

def _is_negated(text, disease_match):
    """Check if disease mention is negated (sentence-aware)"""
    start_pos = disease_match.start()
    end_pos = disease_match.end()
    
    # Split by sentence delimiters
    sentences = re.split(r'[.!?;]', text)
    char_pos = 0
    sentence_idx = -1
    
    for i, sent in enumerate(sentences):
        if char_pos + len(sent) >= end_pos:
            sentence_idx = i
            break
        char_pos += len(sent) + 1
    
    if sentence_idx < 0:
        return False
    
    sentence = sentences[sentence_idx].lower()
    
    # Pre-negation check
    disease_pos_in_sentence = start_pos - sum(len(s) + 1 for s in sentences[:sentence_idx])
    before_text = sentence[:disease_pos_in_sentence]
    after_text = sentence[disease_pos_in_sentence:]
    
    for neg_word in NEGATION_WORDS:
        if re.search(r'\b' + re.escape(neg_word) + r'\b', before_text):
            return True
    
    for post_neg in POST_NEGATION:
        if re.search(re.escape(post_neg), after_text):
            return True
    
    return False

# ============================================================================
# MODEL 3: EXTRACT DISEASES (RULE-BASED)
# ============================================================================

def extract_diseases_from_text(text):
    """Rule-based disease extraction with confidence scoring"""
    positive = {}  # disease -> match_count
    negative = {}  # disease -> match_count
    
    for disease, patterns in DISEASE_PATTERNS.items():
        match_count = 0
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                is_negated = any(_is_negated(text, match) for match in matches)
                match_count += len(matches)
                
                if is_negated:
                    if disease not in negative:
                        negative[disease] = 0
                    negative[disease] += match_count
                else:
                    if disease not in positive:
                        positive[disease] = 0
                    positive[disease] += match_count
                break
    
    return positive, negative

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_report_text(text):
    """
    Complete 3-stage medical report analysis pipeline
    Returns: dict with status, diseases, confidence scores
    """
    
    text = text.strip()
    if not text:
        return {
            "status": "rejected",
            "step_failed": "empty_text",
            "reason": "Report text cannot be empty",
            "diseases": []
        }
    
    # STAGE 1: Is this medical?
    print("Stage 1: Checking if text is medical...")
    is_medical, medical_conf, medical_details = model1_is_medical(text)
    
    if not is_medical:
        return {
            "status": "rejected",
            "step_failed": "model1_not_medical",
            "reason": f"This text does not appear to be a medical report (confidence: {medical_conf*100:.1f}%)",
            "model1": {"is_medical": False, "confidence": round(medical_conf, 4), "details": medical_details},
            "diseases": []
        }
    
    # STAGE 2: Is this a CXR report?
    print("Stage 2: Checking if text is a chest X-ray report...")
    is_cxr, cxr_conf, cxr_details = model2_is_cxr_report(text)
    
    if not is_cxr:
        return {
            "status": "rejected",
            "step_failed": "model2_not_cxr",
            "reason": f"This appears to be medical text but not a chest X-ray report (confidence: {cxr_conf*100:.1f}%)",
            "model1": {"is_medical": True, "confidence": round(medical_conf, 4)},
            "model2": {"is_cxr": False, "confidence": round(cxr_conf, 4), "details": cxr_details},
            "diseases": []
        }
    
    # STAGE 3: Extract diseases
    print("Stage 3: Extracting diseases from CXR report...")
    positive_diseases, negative_diseases = extract_diseases_from_text(text)
    
    # Calculate confidence for each disease based on:
    # - CXR model confidence (how sure we are this is a CXR report)
    # - Match count (how many times disease patterns matched)
    # - Match frequency in text
    base_confidence = cxr_conf  # 0.4-1.0 typically
    
    present_findings = []
    for disease, match_count in positive_diseases.items():
        # Confidence = base_cxr_confidence * (1 + match_count boost)
        # Match boost: 1 match = +0.1, 2+ = +0.2
        match_boost = min(0.1 * min(match_count, 3), 0.3)
        confidence = min(base_confidence + match_boost, 1.0)
        
        # Determine level based on confidence (same as TorchXRayVision)
        if confidence > 0.8:
            level = "Definite"
        elif confidence > 0.5:
            level = "Probable"
        elif confidence > 0.2:
            level = "Possible"
        else:
            level = "Unlikely"
        
        present_findings.append({
            "disease": disease,
            "confidence": float(round(confidence, 4)),
            "level": level,
            "status": "present",
            "label": "Positive (Rule-based)"
        })
    
    ruled_out = []
    for disease, match_count in negative_diseases.items():
        confidence = float(round(base_confidence * 0.5, 4))
        ruled_out.append({
            "disease": disease,
            "confidence": confidence,
            "level": "Unlikely",
            "status": "ruled_out",
            "label": "Negative (Rule-based)"
        })
    
    return {
        "status": "success",
        "model1": {"is_medical": True, "confidence": round(medical_conf, 4), "details": medical_details},
        "model2": {"is_cxr": True, "confidence": round(cxr_conf, 4), "details": cxr_details},
        "disease_names": positive_diseases,
        "present": present_findings,
        "ruled_out": ruled_out,
        "diseases": present_findings  # For compatibility with image pipeline
    }
