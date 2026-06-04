"""
Chest X-Ray Analysis Pipeline (Local Models)
=============================================
Step 1: BiomedCLIP — validate if image is a chest X-ray
Step 2: TorchXRayVision — detect 18 diseases
"""

import torch
import numpy as np
from PIL import Image
from io import BytesIO

# Lazy-loaded model singletons (loaded once, cached in memory)

_biomedclip = None
_biomedclip_preprocess = None
_biomedclip_tokenizer = None

_xrv_model = None
_xrv_transform = None


def _load_biomedclip():
    global _biomedclip, _biomedclip_preprocess, _biomedclip_tokenizer
    if _biomedclip is not None:
        return
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BiomedCLIP on {device}...")
    _biomedclip, _biomedclip_preprocess, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    _biomedclip_tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    _biomedclip = _biomedclip.to(device).eval()
    print("BiomedCLIP loaded.")


def _load_xrv():
    global _xrv_model, _xrv_transform
    if _xrv_model is not None:
        return
    import torchxrayvision as xrv
    import torchvision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading TorchXRayVision on {device}...")
    _xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    _xrv_model = _xrv_model.to(device).eval()
    _xrv_transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    print("TorchXRayVision loaded (18 pathologies).")


# ─── UTILITY: Validate NaN Values ────────────────────────────────────────────

def _is_valid_confidence(value):
    """Check if value is a valid confidence score (0-1 float, not NaN/inf)"""
    if value is None:
        return False
    try:
        fval = float(value)
        if np.isnan(fval) or np.isinf(fval):
            return False
        if fval < 0 or fval > 1:
            return False
        return True
    except (TypeError, ValueError):
        return False

def _sanitize_confidence(value, default=0.05):
    """Convert any value to valid confidence score, or use default"""
    if _is_valid_confidence(value):
        return float(value)
    return default

# ─── STEP 0: Validate Medical Document Image ────────────────────────────────

def is_medical_document(image_bytes):
    """
    Validate if image is NOT an animal, person, or object.
    More lenient - focuses on REJECTING clearly non-medical content.
    Used for report images to reject cat images, selfies, etc.
    Returns (is_valid_report: bool, confidence: float, details: str)
    """
    _load_biomedclip()
    device = next(_biomedclip.parameters()).device

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = _biomedclip_preprocess(image).unsqueeze(0).to(device)

    labels = [
        "this is a screenshot or document with text",
        "this is a medical chart or report",
        "this is a radiology report or findings",
        "this is a handwritten note or clinical notes",
        "this is a medical image or x-ray",
        "this is a photo of an animal like a cat, dog, or pet",
        "this is a photo of a person's face or selfie",
        "this is a photo of a normal everyday object",
        "this is a screenshot of something",
        "this is not a medical or text document",
    ]

    text_tokens = _biomedclip_tokenizer(labels, context_length=256).to(device)

    with torch.no_grad():
        image_features = _biomedclip.encode_image(image_tensor)
        text_features = _biomedclip.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = _biomedclip.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

    probs = logits.squeeze().cpu().tolist()
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_score = probs[top_idx]

    # STRICT REJECTION: Only reject clearly non-medical content
    # Animals, people, and random objects should be rejected
    is_clearly_rejected = (
        "animal" in top_label.lower() or
        "cat" in top_label.lower() or
        "dog" in top_label.lower() or
        "pet" in top_label.lower() or
        "person" in top_label.lower() or
        "selfie" in top_label.lower() or
        "face" in top_label.lower() or
        "everyday object" in top_label.lower()
    )
    
    # If it's clearly an animal/person/object, reject it
    if is_clearly_rejected:
        return False, round(top_score, 4), top_label.replace("this is a ", "").replace("this is ", "")
    
    # Otherwise, allow it (could be medical document, screenshot, or text)
    # Trust that text extraction will verify it's meaningful medical text
    return True, round(top_score, 4), top_label.replace("this is a ", "").replace("this is ", "")

# ─── STEP 1: Validate Chest X-Ray ────────────────────────────────────────────

def is_chest_xray(image_bytes):
    """
    Validate if image is a chest X-ray using BiomedCLIP zero-shot classification.
    Returns (is_valid: bool, confidence: float, details: str)
    """
    _load_biomedclip()
    device = next(_biomedclip.parameters()).device

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = _biomedclip_preprocess(image).unsqueeze(0).to(device)

    labels = [
        "this is a photo of a chest x-ray",
        "this is a photo of a hand x-ray",
        "this is a photo of a brain mri",
        "this is a photo of a knee x-ray",
        "this is a photo of a dental x-ray",
        "this is a photo of a normal person",
        "this is a photo of an animal",
        "this is a photo of a robot or toy",
        "this is a photo of an object or thing",
        "this is a screenshot or document",
        "this is a photo of something that is not medical",
    ]

    text_tokens = _biomedclip_tokenizer(labels, context_length=256).to(device)

    with torch.no_grad():
        image_features = _biomedclip.encode_image(image_tensor)
        text_features = _biomedclip.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = _biomedclip.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

    probs = logits.squeeze().cpu().tolist()
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_score = probs[top_idx]

    is_cxr = "chest x-ray" in top_label
    return is_cxr, round(top_score, 4), top_label.replace("this is a photo of a ", "")


# ─── STEP 2: Disease Detection ────────────────────────────────────────────────

def detect_diseases(image_bytes):
    """
    Detect diseases using TorchXRayVision (18 pathologies).
    Returns (formatted_text: str, findings_list: list)
    """
    _load_xrv()
    device = next(_xrv_model.parameters()).device

    image = Image.open(BytesIO(image_bytes)).convert("L")
    img_array = np.array(image).astype(np.float32)

    if img_array.max() > 1:
        img_array = (img_array / 255.0) * 2 - 1
        img_array = img_array * 1024

    if img_array.ndim == 2:
        img_array = img_array[np.newaxis, ...]

    img_tensor = _xrv_transform(img_array)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = _xrv_model(img_tensor)

    findings = []
    pathologies = _xrv_model.pathologies
    pred_values = preds[0].cpu().tolist()

    indexed = sorted(enumerate(pred_values), key=lambda x: x[1], reverse=True)

    for idx, prob in indexed[:3]:  # TOP 3 ONLY
        if prob > 0.05:
            if prob > 0.8:
                level = "Definite"
            elif prob > 0.5:
                level = "Probable"
            elif prob > 0.2:
                level = "Possible"
            else:
                level = "Unlikely"
            findings.append({
                "disease": pathologies[idx],
                "confidence": round(prob, 4),
                "level": level,
            })

    if not findings:
        return "No significant abnormalities detected. Chest X-ray appears NORMAL.", []

    text_lines = []
    for f in findings:
        pct = f["confidence"] * 100
        if f["confidence"] > 0.5:
            marker = "***"
        elif f["confidence"] > 0.2:
            marker = "**"
        else:
            marker = "*"
        text_lines.append(f'{marker} {f["disease"]}: {f["level"]} ({pct:.1f}%)')

    header = "Disease Detection Results (TorchXRayVision - 18 pathologies):\n"
    header += "(* = Unlikely, ** = Possible, *** = Probable/Definite)\n\n"

    return header + "\n".join(text_lines), findings
