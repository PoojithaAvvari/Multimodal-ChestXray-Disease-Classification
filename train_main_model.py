# src/train_gpu.py
import argparse
import json, os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset_classification_main import IUDataset
from arch_model import MultimodalClassifier
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train_labeled.csv")
    p.add_argument("--val_csv", default="data/test_labeled.csv")
    p.add_argument("--labels_json", default="data/label_names.json")
    p.add_argument("--text_model", default="bert-base-uncased")
    p.add_argument("--image_model", default="resnet18")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--out", default="best_model.pt")
    return p.parse_args()

def compute_metrics(y_true, y_logits):
    from sklearn.metrics import f1_score, roc_auc_score
    y_prob = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    aurocs = []
    try:
        for i in range(y_true.shape[1]):
            aurocs.append(roc_auc_score(y_true[:,i], y_prob[:,i]))
        mean_auroc = float(np.nanmean(aurocs))
    except Exception:
        mean_auroc = None
    return {"f1_macro": float(f1), "auroc": mean_auroc}

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    LABELS = json.load(open(args.labels_json))
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    train_ds = IUDataset(args.train_csv, tokenizer, max_length=args.max_len)
    val_ds   = IUDataset(args.val_csv, tokenizer, max_length=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = MultimodalClassifier(num_labels=len(LABELS),
                                 text_model_name=args.text_model,
                                 image_model=args.image_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')  # we will mask uncertain

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
            imgs = batch['image'].to(device, non_blocking=True)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            uncertain = batch.get('uncertain', torch.zeros_like(labels)).to(device)
            certain_mask = (uncertain == 0).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs, ids, mask)
                loss_mat = loss_fn(logits, labels)
                loss_mat = loss_mat * certain_mask
                loss = loss_mat.sum() / (certain_mask.sum() + 1e-8)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} train_loss: {avg_train_loss:.4f}")

        # validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                imgs = batch['image'].to(device, non_blocking=True)
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu().numpy()
                logits = model(imgs, ids, mask).cpu().numpy()
                all_logits.append(logits)
                all_labels.append(labels)
        all_logits = np.vstack(all_logits)
        all_labels = np.vstack(all_labels)
        metrics = compute_metrics(all_labels, all_logits)
        print("Validation metrics:", metrics)

        if metrics["f1_macro"] > best_val:
            best_val = metrics["f1_macro"]
            torch.save(model.state_dict(), args.out)
            print("Saved best model to", args.out)

    print("Training finished")

if __name__ == "__main__":
    main()
