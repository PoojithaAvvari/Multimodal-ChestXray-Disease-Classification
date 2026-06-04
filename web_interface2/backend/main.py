import os
import sys
import json
import torch
import torch.nn as nn
import timm
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import numpy as np
from dotenv import load_dotenv
from google import genai
import cv2
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add parent directory to path to import 'model'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from arch_model import MultimodalClassifier

# Import the new pipeline (BiomedCLIP + TorchXRayVision)
from pipeline import is_chest_xray, detect_diseases, _sanitize_confidence
# Import text pipeline for report analysis
from text_pipeline import analyze_report_text

load_dotenv()

app = FastAPI(title="Deepfake Med Interface")

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai_client = genai.Client(api_key=gemini_key)
else:
    genai_client = None
    print("WARNING: GEMINI_API_KEY not found in environment. OCR will fail.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GLOBAL MODEL PATHS ---
AUTH_MODEL_PATH = os.path.join(PROJECT_ROOT, "auth_model_best.pt")
MAIN_MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pt")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "label_names.json")

# --- LOAD LABELS ---
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

THRESHOLD = 0.11
# THRESHOLD = 0.94

# ---------------------------
# 1. LOAD AUTH MODEL (GAN Detection)
# ---------------------------
auth_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading Authenticity Model...")
auth_model = timm.create_model("efficientnet_b3", pretrained=False)
auth_model.classifier = nn.Linear(auth_model.classifier.in_features, 1)

auth_checkpoint = torch.load(AUTH_MODEL_PATH, map_location=device, weights_only=False)
if "model_state_dict" in auth_checkpoint:
    auth_model.load_state_dict(auth_checkpoint["model_state_dict"])
    auth_threshold = auth_checkpoint.get("best_threshold", 0.5)
else:
    auth_model.load_state_dict(auth_checkpoint)
    auth_threshold = 0.5

auth_model = auth_model.to(device)
auth_model.eval()

# ---------------------------
# 2. LOAD MAIN MODEL (Disease Classification) — kept for reference, not used in active pipeline
# ---------------------------
main_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading Main Disease Classification Model...")
main_model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
main_model.load_state_dict(torch.load(MAIN_MODEL_PATH, map_location=device, weights_only=False))
main_model.eval()

# Tokenizer lazy-loaded in text_pipeline on first use

# ---------------------------
# 3. NEW PIPELINE MODELS (BiomedCLIP + TorchXRayVision) — lazy-loaded on first request
# ---------------------------
# These are loaded automatically on first call via pipeline.py

# ============================================================================
# HELPER FUNCTIONS FOR ENSEMBLE MERGING
# ============================================================================

def merge_disease_findings(image_findings, report_findings, image_conf, report_conf):
    """
    Confidence-based ensemble: merge image and report findings
    
    Logic:
    - If image_conf > 0.7 and report_conf < 0.5: Trust image only
    - If report_conf > 0.7 and image_conf < 0.5: Trust report only
    - If both confident (>0.6): Merge with averaged confidence
    - Otherwise: Show both with source labels
    """
    # Sanitize input confidences
    image_conf = _sanitize_confidence(image_conf, default=0.5)
    report_conf = _sanitize_confidence(report_conf, default=0.7)
    
    merged = {}
    sources = {}  # Track which source provided each finding
    
    # Add image findings
    for finding in image_findings:
        disease = finding["disease"]
        if disease not in merged:
            merged[disease] = {
                "disease": disease,
                "image_confidence": _sanitize_confidence(finding.get("confidence"), default=0.05),
                "report_confidence": None,
                "level": finding.get("level", "Possible"),
                "sources": ["image"]
            }
        else:
            merged[disease]["sources"].append("image")
    
    # Add/merge report findings
    for finding in report_findings:
        disease = finding["disease"]
        if disease not in merged:
            merged[disease] = {
                "disease": disease,
                "image_confidence": None,
                "report_confidence": _sanitize_confidence(finding.get("confidence"), default=0.5),
                "level": finding.get("level", "Possible"),
                "sources": ["report"]
            }
        else:
            merged[disease]["report_confidence"] = _sanitize_confidence(finding.get("confidence"), default=0.5)
            merged[disease]["sources"].append("report")
    
    # Calculate final confidence and trust decision
    final_findings = []
    for disease, data in merged.items():
        image_conf_val = data["image_confidence"] or 0
        report_conf_val = data["report_confidence"] or 0
        
        # Sanitize values
        image_conf_val = _sanitize_confidence(image_conf_val, default=0)
        report_conf_val = _sanitize_confidence(report_conf_val, default=0)
        
        # Confidence-based decision logic
        if image_conf_val > 0.7 and report_conf_val < 0.5:
            # Trust image only
            final_conf = image_conf_val
            trust_source = "image"
        elif report_conf_val > 0.7 and image_conf_val < 0.5:
            # Trust report only
            final_conf = report_conf_val
            trust_source = "report"
        elif image_conf_val > 0.6 and report_conf_val > 0.6:
            # Both confident: average them
            final_conf = (image_conf_val + report_conf_val) / 2
            trust_source = "both"
        else:
            # Use the higher confidence
            if image_conf_val >= report_conf_val:
                final_conf = image_conf_val
                trust_source = "image_higher"
            else:
                final_conf = report_conf_val
                trust_source = "report_higher"
        
        # Ensure final_conf is valid
        final_conf = _sanitize_confidence(final_conf, default=0.05)
        
        # Determine level based on final confidence
        if final_conf > 0.8:
            level = "Definite"
        elif final_conf > 0.5:
            level = "Probable"
        elif final_conf > 0.2:
            level = "Possible"
        else:
            level = "Unlikely"
        
        final_findings.append({
            "disease": disease,
            "confidence": round(final_conf, 4),
            "level": level,
            "image_confidence": round(image_conf_val, 4) if image_conf_val > 0 else None,
            "report_confidence": round(report_conf_val, 4) if report_conf_val > 0 else None,
            "sources": data["sources"],
            "trust_source": trust_source
        })
    
    # Sort by confidence descending
    final_findings.sort(key=lambda x: x["confidence"], reverse=True)
    return final_findings[:3]  # Return top 3


@app.post("/analyze")
async def analyze(request: Request):
    form_data = await request.form()
    image = form_data.get("image")
    report_image = form_data.get("report_image")
    report_text = form_data.get("report_text")
    
    if not image and not report_image and not report_text:
        return {"error": "Please provide an X-Ray image or a clinical report (image or text)."}

# Handle three cases: image only, report only, or both
    has_valid_image = image and hasattr(image, "read") and image.filename
    has_valid_report_image = report_image and hasattr(report_image, "read") and report_image.filename
    has_valid_report_text = report_text and isinstance(report_text, str) and report_text.strip()
    
    # Pipeline results
    image_result = {"success": False, "findings": None, "error": None, "auth_prob": None, "cxr_conf": None}
    report_result = {"success": False, "findings": None, "error": None}
    
    # ========== PIPELINE 1: Image Analysis ==========
    if has_valid_image:
        try:
            img_bytes = await image.read()
            img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            is_valid_cxr, cxr_confidence, cxr_label = is_chest_xray(img_bytes)
            
            if not is_valid_cxr:
                image_result["error"] = f"Not a chest X-ray. Detected as: {cxr_label}"
            else:
                # Check authenticity
                auth_img_t = auth_tf(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    auth_prob = torch.sigmoid(auth_model(auth_img_t)).item()
                
                is_fake = float(auth_prob) > float(auth_threshold)
                if is_fake:
                    print(f"Deepfake detected with prob {auth_prob}")
                    
                    # Generate CAM
                    explanation_image = None
                    try:
                        target_layers = [auth_model.conv_head]
                        cam = GradCAM(model=auth_model, target_layers=target_layers)
                        
                        targets = [ClassifierOutputTarget(0)]
                        grayscale_cam = cam(input_tensor=auth_img_t, targets=targets)
                        grayscale_cam = grayscale_cam[0, :]
                        
                        img_pil_resized = img_pil.resize((300, 300))
                        rgb_img = np.float32(img_pil_resized) / 255
                        cam_image_np = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                        
                        cam_pil = Image.fromarray(cam_image_np)
                        buffered = BytesIO()
                        cam_pil.save(buffered, format="JPEG")
                        cam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        explanation_image = f"data:image/jpeg;base64,{cam_b64}"
                    except Exception as cam_e:
                        print(f"Failed to generate CAM for explainability: {cam_e}")

                    # Return immediately to stop prediction and show modal on frontend
                    return {
                        "status": "fake",
                        "auth_probability": float(auth_prob),
                        "explanation_image": explanation_image
                    }
                else:
                    # Get diseases
                    diseases_text, disease_findings = detect_diseases(img_bytes)
                    
                    # Validate and sanitize findings - robust NaN handling
                    for finding in disease_findings:
                        # Ensure confidence is valid
                        conf = finding.get("confidence", 0.05)
                        finding["confidence"] = _sanitize_confidence(conf, default=0.05)
                        
                        # Ensure level is valid
                        if finding.get("level") not in ["Definite", "Probable", "Possible", "Unlikely"]:
                            finding["level"] = "Possible"
                    
                    if disease_findings:
                        image_result["success"] = True
                        image_result["findings"] = disease_findings
                        image_result["auth_prob"] = auth_prob
                        image_result["cxr_conf"] = _sanitize_confidence(cxr_confidence, default=0.5)
                    else:
                        image_result["error"] = "No diseases detected in X-ray image"
        except Exception as e:
            print(f"Image pipeline error: {e}")
            image_result["error"] = str(e)
    
    # ========== PIPELINE 2: Report Analysis ==========
    if has_valid_report_image or has_valid_report_text:
        try:
            extracted = None
            
            # If we have report image, extract text from it via Gemini OCR
            if has_valid_report_image:
                if not genai_client:
                    report_result["error"] = "GEMINI_API_KEY not configured"
                else:
                    report_bytes = await report_image.read()
                    report_pil = Image.open(BytesIO(report_bytes)).convert("RGB")
                    
                    # Extract text via Gemini OCR - works on any image
                    print("Extracting text from report image via Gemini OCR...")
                    response = genai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            report_pil,
                            """Extract only clinical X-ray findings from this report image. Remove all non-medical info. 
                            Output plain text only. If no findings, return: "no relevant xray findings" """
                        ]
                    )
                    extracted = response.text.strip() if response.text else ""
                    print(f"Extracted Report Text: {extracted[:150]}...")
            
            # If we have direct text input, use it as is
            elif has_valid_report_text:
                extracted = report_text.strip()
                print(f"Using direct report text input: {extracted[:150]}...")
            
            # Analyze extracted or provided text
            if extracted and extracted != "no relevant xray findings":
                print(f"\n=== ANALYZING REPORT TEXT ===\nText: {extracted[:200]}...\n")
                
                analysis_result = analyze_report_text(extracted)
                
                if analysis_result["status"] == "success":
                    positive_findings = analysis_result.get("present", [])
                    
                    # Validate and sanitize confidence scores - robust NaN handling
                    for finding in positive_findings:
                        conf = finding.get("confidence", 0.5)
                        finding["confidence"] = _sanitize_confidence(conf, default=0.5)
                        
                        # Ensure level is valid
                        if finding.get("level") not in ["Definite", "Probable", "Possible", "Unlikely"]:
                            finding["level"] = "Possible"
                    
                    if positive_findings:
                        report_result["success"] = True
                        report_result["findings"] = positive_findings
                        print(f"Report analysis found {len(positive_findings)} diseases")
                    else:
                        report_result["error"] = "No diseases detected in report text analysis"
                else:
                    report_result["error"] = analysis_result.get("reason", "Failed to analyze report text")
            else:
                report_result["error"] = "No relevant report content provided or extracted"
        except Exception as e:
            print(f"Report pipeline error: {e}")
            report_result["error"] = str(e)
    
    # ========== DECISION LOGIC ==========
    # Both pipelines successful: merge them
    if image_result["success"] and report_result["success"]:
        print("\n=== MERGING BOTH PIPELINES ===")
        
        img_conf = image_result["cxr_conf"] or 0.5
        report_conf = 0.7
        
        merged_findings = merge_disease_findings(
            image_result["findings"], 
            report_result["findings"], 
            img_conf, 
            report_conf
        )
        
        return {
            "status": "success",
            "mode": "dual_pipeline",
            "image_pipeline": {
                "status": "success",
                "auth_probability": round(float(image_result["auth_prob"]), 4),
                "cxr_confidence": _sanitize_confidence(image_result["cxr_conf"], default=0.5)
            },
            "report_pipeline": {
                "status": "success",
                "extraction_confidence": _sanitize_confidence(report_conf, default=0.7)
            },
            "disease_findings": merged_findings,
            "merge_strategy": "Confidence-averaged ensemble"
        }
    
    # Only image successful
    elif image_result["success"]:
        print("\n=== IMAGE ONLY (Report failed) ===")
        
        return {
            "status": "success",
            "mode": "image_only",
            "image_pipeline": {
                "status": "success",
                "auth_probability": round(float(image_result["auth_prob"]), 4),
                "cxr_confidence": _sanitize_confidence(image_result["cxr_conf"], default=0.5)
            },
            "report_pipeline": {
                "status": "failed",
                "error": report_result["error"]
            },
            "disease_findings": image_result["findings"],
            "message": f"Report analysis failed: {report_result['error']}. Using X-ray analysis only."
        }
    
    # Only report successful
    elif report_result["success"]:
        print("\n=== REPORT ONLY (Image failed) ===")
        
        return {
            "status": "success",
            "mode": "report_only",
            "image_pipeline": {
                "status": "failed",
                "error": image_result["error"]
            },
            "report_pipeline": {
                "status": "success",
                "extraction_confidence": 0.7
            },
            "disease_findings": report_result["findings"],
            "message": f"X-ray analysis failed: {image_result['error']}. Using report analysis only."
        }
    
    # Both failed
    else:
        return {
            "status": "rejected",
            "image_pipeline": {
                "status": "failed",
                "error": image_result["error"]
            },
            "report_pipeline": {
                "status": "failed",
                "error": report_result["error"]
            },
            "reason": "Both pipelines failed. No valid diagnosis could be generated."
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
