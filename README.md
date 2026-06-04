# 🏥 Deepfake Medical Image Detection & Disease Classification System

## 📋 Project Overview

A comprehensive medical imaging analysis system that:
- 🔍 **Detects AI-generated (deepfake) X-ray images** using EfficientNet-B3
- ✓ **Validates chest X-ray modality** using BiomedCLIP vision-language model
- 🏥 **Detects 18 chest pathologies** using TorchXRayVision (DenseNet121)
- 📊 **Provides clinical-grade analysis** with confidence scores

---

## 🚀 Quick Start (3 Steps)

### Step 1: Activate Virtual Environment & Install Dependencies
```powershell
cd "c:\Users\pooji\Desktop\majoprojec t"
& ".\diffvenv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

### Step 2: Start Backend API (Terminal 1)
```powershell
cd "c:\Users\pooji\Desktop\majoprojec t\web_interface2\backend"
& ..\..\diffvenv\Scripts\Activate.ps1
python main.py
```
✅ Backend will run on: **http://127.0.0.1:8000**

### Step 3: Start Frontend UI (Terminal 2)
```powershell
cd "c:\Users\pooji\Desktop\majoprojec t\web_interface2\frontend"
npm install
npm run dev
```
✅ Frontend will run on: **http://localhost:5173**

---

## 🎯 Usage

### Upload & Analyze X-ray Image

1. Open browser: **http://localhost:5173**
2. Click "Upload X-ray Image"
3. Select a chest X-ray file (JPG, PNG, etc.)
4. View results:

```
✓ Image Status: REAL (Authenticity: 99.2%)
✓ Chest X-ray Validated: Yes (Confidence: 98.5%)
✓ Top 3 Diseases Detected:
   *** Pneumonia: Definite (87.5%)
   ** Consolidation: Probable (62.3%)
   * Atelectasis: Possible (28.1%)
```

---

## 📁 Project Structure

```
majoprojec t/
├── web_interface2/
│   ├── backend/
│   │   ├── main.py                 # FastAPI server
│   │   ├── pipeline.py             # BiomedCLIP + TorchXRayVision models
│   │   └── requirements.txt        # Backend dependencies
│   └── frontend/
│       ├── src/                    # React components
│       ├── package.json            # Frontend dependencies
│       └── vite.config.js          # Vite configuration
├── arch_model.py                   # MultimodalClassifier (disease classification)
├── auth_model_best.pt              # EfficientNet-B3 (deepfake detector)
├── best_model.pt                   # Main disease classifier
├── data/                           # Dataset & labels
└── diffvenv/                       # Python virtual environment
```

---

## 🧠 AI Models Used

### 1. **EfficientNet-B3** (Authenticity Detection)
- **Purpose**: Detect AI-generated vs. real X-rays
- **Input**: 300×300 RGB images
- **Output**: Probability score (0-1)
- **Threshold**: > 0.5 = Likely Fake

### 2. **BiomedCLIP** (X-ray Validation)
- **Purpose**: Confirm image is a chest X-ray
- **Model**: Vision-language model (microsoft/BiomedCLIP-PubMedBERT)
- **Input**: RGB image + text labels
- **Output**: Classification + confidence

### 3. **TorchXRayVision (DenseNet121)** (Disease Detection)
- **Purpose**: Identify 18 chest pathologies
- **Model**: Pre-trained on CheXpert, MIMIC-CXR
- **Detects**: 
  - Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion
  - Emphysema, Fibrosis, Infiltration, Mass, Nodule
  - Pleural Thickening, Pneumonia, Pneumothorax, and more
- **Output**: Top 3 predictions with confidence scores

---

## 🔧 API Endpoints

### POST `/analyze`
Upload and analyze a medical image

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "image=@xray.jpg"
```

**Response:**
```json
{
  "status": "success",
  "auth_probability": 0.05,
  "chest_xray_confidence": 0.985,
  "detected_label": "chest x-ray",
  "diseases": "Disease Detection Results...",
  "disease_findings": [
    {
      "disease": "Pneumonia",
      "confidence": 0.875,
      "level": "Definite"
    }
  ]
}
```

### GET `/docs`
Interactive Swagger API documentation: **http://127.0.0.1:8000/docs**

---

## ⚙️ Configuration

### Modify Top-N Diseases (Currently: Top 3)
**File**: `web_interface2/backend/pipeline.py` (Line ~133)

Change:
```python
for idx, prob in indexed[:3]:  # Change 3 to desired number
```

### Adjust Authenticity Threshold
**File**: `web_interface2/backend/main.py` (Line ~60)

```python
THRESHOLD = 0.11  # Adjust sensitivity (lower = more sensitive to fakes)
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────┐
│    React/Vite Frontend (Port 5173)          │
│  - Web UI for image upload                  │
│  - Real-time results display                │
└────────────────┬────────────────────────────┘
                 │ HTTP/REST API
                 ▼
┌─────────────────────────────────────────────┐
│    FastAPI Backend (Port 8000)              │
│  ┌─────────────────────────────────────┐    │
│  │ Authenticity Model (EfficientNet-B3)│    │
│  │ → Real vs. AI-generated Detection    │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ Chest X-Ray Validator (BiomedCLIP)  │    │
│  │ → Modality Confirmation             │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ Disease Detector (TorchXRayVision)  │    │
│  │ → Top 3 Pathology Detection         │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## 🔑 Key Features

✅ **End-to-End Pipeline**
- Single image upload → Complete analysis

✅ **Multiple AI Models**
- Leverages 3 specialized medical AI models

✅ **Real-time Processing**
- Fast inference (CPU compatible)

✅ **Production Ready**
- CORS enabled, error handling included

✅ **Clinical Confidence Levels**
- Definite (>80%), Probable (50-80%), Possible (20-50%), Unlikely (<20%)

---

## 📈 Performance

| Model | Framework | Speed | Device |
|-------|-----------|-------|--------|
| EfficientNet-B3 | PyTorch | ~500ms | CPU/GPU |
| BiomedCLIP | PyTorch | ~1000ms | CPU/GPU |
| TorchXRayVision | PyTorch | ~800ms | CPU/GPU |
| **Total** | **Async Pipeline** | **~2.3s** | **CPU** |

---

## 🛠️ Troubleshooting

### Backend Won't Start
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Frontend Build Issues
```powershell
cd web_interface2/frontend
rm -r node_modules package-lock.json
npm install
npm run dev
```

### Model Download Failed
```powershell
# TorchXRayVision models auto-download on first run
# If stuck, download manually:
cd %USERPROFILE%\.torchxrayvision\models_data
# Monitor for completion (few GB)
```

---

## 📝 Environment Requirements

- **Python**: 3.10+
- **PyTorch**: 2.4.1+ (CPU/CUDA supported)
- **Node.js**: 18+ (for frontend)
- **RAM**: 8GB minimum
- **Storage**: 5GB for models

---

## 🚀 Deployment

### Run Both Services with Single Command
```powershell
# Backend
Start-Process powershell -ArgumentList "cd 'c:\Users\pooji\Desktop\majoprojec t\web_interface2\backend'; & ..\..\diffvenv\Scripts\Activate.ps1; python main.py"

# Frontend (separate terminal)
Start-Process powershell -ArgumentList "cd 'c:\Users\pooji\Desktop\majoprojec t\web_interface2\frontend'; npm run dev"
```

### Access Points
- **UI**: http://localhost:5173
- **API**: http://127.0.0.1:8000
- **API Docs**: http://127.0.0.1:8000/docs
- **ReDoc Docs**: http://127.0.0.1:8000/redoc

---

## 📚 References

- **BiomedCLIP**: [microsoft/BiomedCLIP-PubMedBERT](https://github.com/microsoft/BiomedCLIP-PubMedBERT)
- **TorchXRayVision**: [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision)
- **EfficientNet**: [pytorch/vision](https://github.com/pytorch/vision)

---

## 📄 License & Credits

This system combines state-of-the-art medical AI models for clinical decision support.

**⚠️ DISCLAIMER**: This tool is for educational and research purposes. Always consult qualified radiologists for diagnostic interpretation.

---

**Last Updated**: April 8, 2026
