
# src/test_infer.py
import torch, json
from PIL import Image
from transformers import AutoTokenizer
from arch_model import MultimodalClassifier
from dataset_classification_main import image_transform
import numpy as np
from auth_inference import authenticity_check

LABELS = json.load(open("data/label_names.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.2  # from eval.py best threshold

model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# image_path = "C:\\Users\\pooji\\Downloads\\archive\\images\\images_normalized\\64_IM-2218-4004.dcm.png" //0.38
# image_path = r"C:\Users\pooji\Desktop\t.png" //0.35
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder1\3925_IM-1999-1003002.dcm.png"
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder\1000_IM-0003-2001.dcm.png" //0.75
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays\10_IM-0002-1001.dcm.png" //0.377


# image_path = "C:\\Users\\pooji\\Downloads\\archive\\images\\images_normalized\\64_IM-2218-4004.dcm.png" //correct
# image_path = r"C:\Users\pooji\Desktop\t.png" //detectimg as real-
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder1\3925_IM-1999-1003002.dcm.png"//correctly detect as fake
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder\1000_IM-0003-2001.dcm.png" //correctly detect as fake
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays\10_IM-0002-1001.dcm.png" //detectimg as real


# image_path = "C:\\Users\\pooji\\Downloads\\archive\\images\\images_normalized\\64_IM-2218-4004.dcm.png" -ok
# image_path = r"C:\Users\pooji\Desktop\t.png" //DETECTING AS REAL-WRONG
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder1\3925_IM-1999-1003002.dcm.png"//correctly detect as fake
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder\1000_IM-0003-2001.dcm.png" //correctly detect as fake
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays\10_IM-0002-1001.dcm.png" 
# image_path=r"C:\Users\pooji\Downloads\ChatGPT Image Feb 24, 2026, 11_59_21 AM.png"//DETECTING AS REAL-WRONG
# image_path=r"archive/images/images_normalized\118_IM-0123-1001.dcm.png"
# image_path=r"fake_autoencoder1\3725_IM-1861-1001.dcm.png"


# image_path = "C:\\Users\\pooji\\Downloads\\archive\\images\\images_normalized\\64_IM-2218-4004.dcm.png" -ok
# image_path = r"C:\Users\pooji\Desktop\t.png" 
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder1\3925_IM-1999-1003002.dcm.png"//correctly detect as fake
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder\1000_IM-0003-2001.dcm.png" //correctly detect as fake
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays\10_IM-0002-1001.dcm.png" 
# image_path=r"C:\Users\pooji\Downloads\ChatGPT Image Feb 24, 2026, 11_59_21 AM.png"
# image_path=r"archive/images/images_normalized\118_IM-0123-1001.dcm.png"
# image_path=r"fake_autoencoder1\3725_IM-1861-1001.dcm.png"
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\archive\images\images_normalized\11_IM-0067-1001.dcm.png"
# image_path=r"C:\Users\pooji\Desktop\majoprojec t\archive\images\images_normalized\5_IM-2117-1004003.dcm.png"


# ---------------------------
# Authenticity Check
# ---------------------------
label, auth_prob = authenticity_check(image_path)

print("\n=== Authenticity Check ===")
print(f"AI-generated probability: {auth_prob:.4f}")

if label == 1:
    print("⚠ AI-generated image detected. Diagnosis blocked.")
    exit()
else:
    print("✅ Real image detected. Proceeding to disease classification.\n")

# 1) Load image
img = Image.open(image_path).convert("RGB")
img_t = image_transform(img).unsqueeze(0).to(device)

# 2) Text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
txt = (
    # "A XXXX XXXX lung volumes. Lungs are clear without focal airspace disease. No pleural effusions or pneumothoraces. cardiomegaly. Degenerative changes in the spine.,Cardiomegaly with low lung volumes which are grossly clear.,"
    "The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. There is no pneumothorax or pleural effusion. There are no focal areas of consolidation. Cholecystectomy clips are present. Small T-spine osteophytes. There is biapical pleural thickening, unchanged from prior. Mildly hyperexpanded lungs."
)
inputs = tokenizer(txt, padding="max_length", max_length=128,
                   truncation=True, return_tensors="pt")
ids  = inputs["input_ids"].to(device)
mask = inputs["attention_mask"].to(device)

# 3) Model inference
with torch.no_grad():
    logits = model(img_t, ids, mask)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

print("=== All probabilities ===")
for label, p in zip(LABELS, probs):
    print(f"{label:14s} {p:.4f}")

# 4) Threshold-based multi-label prediction
positive_labels = [label for label, p in zip(LABELS, probs) if p >= THRESHOLD]

print(f"\n=== Threshold-based predictions (th = {THRESHOLD}) ===")
if positive_labels:
    for lab in positive_labels:
        idx = LABELS.index(lab)
        print(f"{lab:14s} {probs[idx]:.4f}  (PREDICTED)")
else:
    print("No label crosses threshold -> no strong abnormality predicted.")

# 5) Top-1 and Top-3 (for display)
top1_idx = int(np.argmax(probs))
top1_label = LABELS[top1_idx]
top1_prob = float(probs[top1_idx])

print("\n=== Top-1 (argmax) ===")
print(f"{top1_label:14s} {top1_prob:.4f}")
if top1_prob < 0.2:
    print("⚠ Low-confidence prediction (max prob < 0.2).")

topk = 3
topk_idx = np.argsort(probs)[::-1][:topk]
print(f"\n=== Top-{topk} ===")
for i in topk_idx:
    print(f"{LABELS[i]:14s} {probs[i]:.4f}")
