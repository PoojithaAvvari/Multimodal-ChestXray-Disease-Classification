import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from arch_model import MultimodalClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train_labeled.csv")
    p.add_argument("--val_csv", default="data/test_labeled.csv")
    p.add_argument("--labels_json", default="data/label_names.json")
    p.add_argument("--text_model", default="bert-base-uncased")
    p.add_argument("--image_model", default="resnet18")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--lr_classifier", type=float, default=1e-3)
    p.add_argument("--lr_encoders", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--out", default="best_model.pt") 
    return p.parse_args()

class FocalLoss(nn.Module):
    """
    Focal Loss solves 'Predicting None for most cases' (class imbalance).
    It focuses heavily on rare diseases without blowing up regular probabilities.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, mask):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        F_loss = F_loss * mask
        return F_loss.sum() / (mask.sum() + 1e-8)


class MedicalAugmentedDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=128, is_train=False):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        text = ""
        if 'impression' in row and isinstance(row['impression'], str) and row['impression'].strip():
            text = row['impression']
        elif 'findings' in row and isinstance(row['findings'], str) and row['findings'].strip():
            text = row['findings']

        # ==========================================
        # MODALITY DROPOUT (TEXT MASKING)
        # ==========================================
        # 50% of the time, hide the text so it can't cheat.
        if self.is_train and np.random.rand() < 0.5:
            text = "" # Completely blind the text encoder
            
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )

        labels_vec = ast.literal_eval(row['labels_vec']) if isinstance(row['labels_vec'], str) else row['labels_vec']
        uncertain_vec = ast.literal_eval(row['uncertain_vec']) if isinstance(row['uncertain_vec'], str) else row['uncertain_vec']

        return {
            "image": image,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels_vec, dtype=torch.float32),
            "uncertain": torch.tensor(uncertain_vec, dtype=torch.float32)
        }

def compute_metrics(y_true, y_logits):
    from sklearn.metrics import f1_score
    y_prob = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {"f1_macro": float(f1)}

def get_optimizer(model, args, freeze_encoders=False):
    if freeze_encoders:
        print("❄️ STAGE 1: Freezing Encoders (BERT/ResNet). Warming up Classification Head...")
        for param in model.cnn.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        return torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_classifier)
    else:
        print("🔥 STAGE 2: Unfreezing entire model for deep fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True
            
        optimizer_grouped_parameters = [
            {"params": model.cnn.parameters(), "lr": args.lr_encoders},
            {"params": model.text_encoder.parameters(), "lr": args.lr_encoders},
            {"params": model.classifier.parameters(), "lr": args.lr_classifier}
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    LABELS = json.load(open(args.labels_json))
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading datasets with Modality Dropout enabled for training...")
    train_ds = MedicalAugmentedDataset(args.train_csv, tokenizer, train_transform, max_length=args.max_len, is_train=True)
    val_ds   = MedicalAugmentedDataset(args.val_csv, tokenizer, val_transform, max_length=args.max_len, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MultimodalClassifier(num_labels=len(LABELS), text_model_name=args.text_model, image_model=args.image_model).to(device)

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = -1.0
    optimizer = None

    for epoch in range(1, args.epochs + 1):
        
        if epoch == 1:
            optimizer = get_optimizer(model, args, freeze_encoders=True)
        elif epoch == args.warmup_epochs + 1:
            optimizer = get_optimizer(model, args, freeze_encoders=False)
            
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.epochs}"):
            imgs = batch['image'].to(device, non_blocking=True)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            uncertain = batch.get('uncertain', torch.zeros_like(labels)).to(device)
            certain_mask = (uncertain == 0).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs, ids, mask)
                loss = loss_fn(logits, labels, certain_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} train_loss: {avg_train_loss:.4f}")

        # Validation
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
            print(f"🎉 New Best Model Found! Saved to {args.out}")

    print("Training finished. Final Best F1:", best_val)

if __name__ == "__main__":
    main()
