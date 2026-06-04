# src/confusion_matrix.py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dataset_classification_main import IUDataset
from arch_model import MultimodalClassifier

TEST_CSV = "data/test_labeled.csv"
LABELS_JSON = "data/label_names.json"
MODEL_WEIGHTS = "best_model.pt"
THRESHOLD = 0.20  # from eval.py

SAVE_DIR = "confusion_matrices"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_everything():
    label_names = json.load(open(LABELS_JSON))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds = IUDataset(TEST_CSV, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MultimodalClassifier(num_labels=len(label_names)).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    return label_names, loader, model, device

def collect_predictions(model, loader, device):
    all_logits = []
    all_labels = []

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
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= THRESHOLD).astype(int)

    return all_labels, preds

def plot_confusion_matrix(cm, label_name):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(f"Confusion Matrix - {label_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    outfile = os.path.join(SAVE_DIR, f"{label_name}_cm.png")
    plt.savefig(outfile)
    plt.close()
    return outfile

def main():
    label_names, loader, model, device = load_everything()
    y_true, y_pred = collect_predictions(model, loader, device)

    print("\nGenerating Confusion Matrices...\n")

    for i, name in enumerate(label_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        path = plot_confusion_matrix(cm, name)
        print(f"Saved: {path}")

    print("\nAll confusion matrices saved in:", SAVE_DIR)

if __name__ == "__main__":
    main()
