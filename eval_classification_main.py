# # src/eval.py
# import json, torch, numpy as np
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from dataset import IUDataset
# from model import MultimodalClassifier
# from sklearn.metrics import f1_score, roc_auc_score

# def main():
#     LABELS = json.load(open("data/label_names.json"))
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#     ds = IUDataset("data/test_labeled.csv", tokenizer, max_length=128)
#     loader = DataLoader(ds, batch_size=8, num_workers=2)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultimodalClassifier(num_labels=len(LABELS)).to(device)
#     model.load_state_dict(torch.load("best_model.pt", map_location=device))
#     model.eval()

#     all_logits, all_labels = [], []

#     with torch.no_grad():
#         for batch in loader:
#             imgs = batch['image'].to(device)
#             ids  = batch['input_ids'].to(device)
#             mask = batch['attention_mask'].to(device)

#             logits = model(imgs, ids, mask).cpu().numpy()
#             labels = batch['labels'].numpy()

#             all_logits.append(logits)
#             all_labels.append(labels)

#     all_logits = np.vstack(all_logits)
#     all_labels = np.vstack(all_labels)
#     probs = 1/(1+np.exp(-all_logits))
#     preds = (probs >= 0.5).astype(int)

#     print("Macro F1:", f1_score(all_labels, preds, average="macro"))

#     try:
#         for i, lab in enumerate(LABELS):
#             auc = roc_auc_score(all_labels[:, i], probs[:, i])
#             print(f"{lab}: AUROC={auc:.4f}")
#     except:
#         print("AUROC calculation error")

# if __name__ == "__main__":
#     main()


# src/eval.py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score

from dataset_classification_main import IUDataset
from arch_model import MultimodalClassifier

TEST_CSV = "data/test_labeled.csv"
LABELS_JSON = "data/label_names.json"
TEXT_MODEL = "bert-base-uncased"
BATCH_SIZE = 8
MAX_LEN = 128
MODEL_WEIGHTS = "best_model.pt"

def load_data_and_model():
    # labels
    label_names = json.load(open(LABELS_JSON))

    # tokenizer + dataset + loader
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    ds = IUDataset(TEST_CSV, tokenizer, max_length=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model
    model = MultimodalClassifier(num_labels=len(label_names)).to(device)
    state = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return label_names, loader, model, device

def collect_logits_and_labels(loader, model, device):
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
    return all_logits, all_labels

def evaluate_thresholds(label_names, y_true, y_logits):
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))  # sigmoid
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    print("\n=== Threshold Tuning ===")
    best_macro_f1 = -1.0
    best_th = None

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        print(f"\n--- Threshold = {th:.2f} ---")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Micro F1: {micro_f1:.4f}")

        # per-label F1
        for i, label in enumerate(label_names):
            f1_i = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
            print(f"  {label:15s} F1 = {f1_i:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_th = th

    print("\n=== Best threshold by Macro F1 ===")
    print(f"Best threshold: {best_th:.2f}  |  Macro F1: {best_macro_f1:.4f}")
    return best_th, best_macro_f1

def evaluate_auroc(label_names, y_true, y_logits):
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    print("\n=== AUROC per label ===")
    aucs = []
    for i, label in enumerate(label_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            aucs.append(auc)
            print(f"{label:15s} AUROC = {auc:.4f}")
        except ValueError:
            # happens if only one class present in y_true[:, i]
            print(f"{label:15s} AUROC = N/A (only one class present)")
    if aucs:
        print(f"\nMean AUROC: {np.mean(aucs):.4f}")

def main():
    label_names, loader, model, device = load_data_and_model()
    y_logits, y_true = None, None

    # collect predictions
    logits, labels = collect_logits_and_labels(loader, model, device)
    y_logits = logits
    y_true = labels

    # threshold tuning
    best_th, best_f1 = evaluate_thresholds(label_names, y_true, y_logits)

    # AUROC
    evaluate_auroc(label_names, y_true, y_logits)

if __name__ == "__main__":
    main()
