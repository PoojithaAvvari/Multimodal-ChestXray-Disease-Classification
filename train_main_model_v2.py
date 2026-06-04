import argparse
import json
import torch
import numpy as np
import pandas as pd
import ast
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# Import your existing Multimodal model
from arch_model import MultimodalClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train_labeled.csv")
    p.add_argument("--val_csv", default="data/test_labeled.csv")
    p.add_argument("--labels_json", default="data/label_names.json")
    p.add_argument("--text_model", default="bert-base-uncased")
    p.add_argument("--image_model", default="resnet18")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8) # Increased slightly for warmup
    p.add_argument("--warmup_epochs", type=int, default=2) # epochs to keep encoders frozen
    p.add_argument("--lr_classifier", type=float, default=1e-3)
    p.add_argument("--lr_encoders", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--out", default="best_model_new_script.pt") # Output path requested
    return p.parse_args()

# ==========================================
# CUSTOM DATASET FOR AUGMENTATION (STEP 3)
# ==========================================
class AugmentedIUDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform, max_length=128):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

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

# ==========================================
# CLASS IMBALANCE WEIGHTING (STEP 1)
# ==========================================
def compute_pos_weights(dataset, num_labels, device):
    print("⚖️ Computing class imbalance weights for Focal/Weighted Loss...")
    label_counts = torch.zeros(num_labels)
    total_samples = len(dataset)
    
    for idx in range(total_samples):
        row = dataset.df.iloc[idx]
        labels_vec = ast.literal_eval(row['labels_vec']) if isinstance(row['labels_vec'], str) else row['labels_vec']
        label_counts += torch.tensor(labels_vec)
        
    # pos_weight = negative_samples / positive_samples
    pos_weights = (total_samples - label_counts) / (label_counts + 1e-5)
    
    # Cap the weight to prevent gradient explosions on extremely rare diseases
    pos_weights = torch.clamp(pos_weights, max=20.0) 
    return pos_weights.to(device)


def compute_metrics(y_true, y_logits):
    from sklearn.metrics import f1_score
    y_prob = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {"f1_macro": float(f1)}

# ==========================================
# DIFFERENTIAL LR & STAGED FREEZING (STEP 2 & 4)
# ==========================================
def get_optimizer(model, args, freeze_encoders=False):
    if freeze_encoders:
        print("\n❄️ STAGE 1: Freezing Encoders (BERT & ResNet). WARMING UP classifier head...")
        for param in model.cnn.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        return torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_classifier)
    else:
        print("\n🔥 STAGE 2: Unfreezing entire model for deep fine-tuning (Differential LRs)...")
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
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_ds = AugmentedIUDataset(args.train_csv, tokenizer, train_transform, max_length=args.max_len)
    val_ds   = AugmentedIUDataset(args.val_csv, tokenizer, val_transform, max_length=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MultimodalClassifier(num_labels=len(LABELS), text_model_name=args.text_model, image_model=args.image_model).to(device)

    # Implement Class Imbalance Loss
    pos_weights = compute_pos_weights(train_ds, len(LABELS), device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
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
                loss_mat = loss_fn(logits, labels)
                loss_mat = loss_mat * certain_mask
                loss = loss_mat.sum() / (certain_mask.sum() + 1e-8)

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

    print("Training completely finished. Final Best F1:", best_val)

if __name__ == "__main__":
    main()
