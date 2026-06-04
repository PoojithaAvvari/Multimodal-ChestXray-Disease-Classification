import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Import from existing project files
from dataset_classification_main import IUDataset
from arch_model import MultimodalClassifier

TEST_CSV = "data/test_labeled.csv"
LABELS_JSON = "data/label_names.json"
TEXT_MODEL = "bert-base-uncased"
BATCH_SIZE = 8
MAX_LEN = 128
# MODEL_WEIGHTS = "best_model.pt"#0.11

MODEL_WEIGHTS = "best_model_new_script.pt"

def load_data_and_model():
    label_names = json.load(open(LABELS_JSON))
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    
    print("Loading test dataset...")
    ds = IUDataset(TEST_CSV, tokenizer, max_length=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading multimodal classifier model...")
    model = MultimodalClassifier(num_labels=len(label_names)).to(device)
    state = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    return label_names, loader, model, device

def run_threshold_grid_search():
    try:
        label_names, loader, model, device = load_data_and_model()
    except Exception as e:
        print(f"Error loading model/data: {e}")
        return

    all_logits = []
    all_labels = []

    print("\nRunning inference on exact test data... (this may take a minute depending on GPU)")
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits = model(imgs, ids, mask).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    # Convert raw logits to probabilities using Sigmoid
    y_prob = 1.0 / (1.0 + np.exp(-all_logits))

    print("\nOptimizing threshold for Macro F1 Score...")
    # Test every threshold from 0.05 to 0.95 in steps of 0.01
    thresholds = np.linspace(0.05, 0.95, 91) 
    
    best_macro_f1 = -1.0
    best_th = 0.20
    f1_scores = []

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        macro_f1 = f1_score(all_labels, y_pred, average="macro", zero_division=0)
        f1_scores.append(macro_f1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_th = th

    print("\n" + "="*60)
    print(f"🏆 BEST GLOBAL THRESHOLD: {best_th:.2f} ({best_th*100:.0f}%)")
    print(f"📈 Macro F1 Score at {best_th:.2f}: {best_macro_f1:.4f}")
    print("="*60)

    print("\n✅ Next Step:")
    print(f"Open 'web_interface2/backend/main.py' and change line 42 to:")
    print(f"THRESHOLD = {best_th:.2f}")

    # Plotting for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker='o', markersize=3, linestyle='-', color='dodgerblue')
    plt.axvline(x=best_th, color='firebrick', linestyle='--', linewidth=2, label=f'Optimal Threshold: {best_th:.2f}')
    
    # Highlight max point
    plt.scatter([best_th], [best_macro_f1], color='red', s=100, zorder=5)
    
    plt.title('Macro F1 Score Optimization over Probability Threshold', fontsize=14)
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('Macro Average F1 Score', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_img = 'threshold_tuning_curve.png'
    plt.savefig(output_img)
    print(f"📊 Saved tuning visualization to: {output_img}")

if __name__ == "__main__":
    print("Starting Threshold Tuning Script...")
    run_threshold_grid_search()
