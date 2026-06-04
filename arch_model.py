# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

class MultimodalClassifier(nn.Module):
    def __init__(self, num_labels, text_model_name="bert-base-uncased", image_model="resnet18", dropout=0.3):
        super().__init__()

        # -------------------------
        # Image Encoder (ResNet18)
        # -------------------------
        if image_model == "resnet18":
            res = models.resnet18(pretrained=True)
            self.image_feat_dim = res.fc.in_features
            self.cnn = nn.Sequential(*list(res.children())[:-1])  # remove FC
        else:
            raise ValueError("Only resnet18 supported for now")

        # -------------------------
        # Text Encoder (BERT)
        # -------------------------
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_feat_dim = self.text_encoder.config.hidden_size

        # -------------------------
        # Fusion + Classifier
        # -------------------------
        fusion_dim = self.image_feat_dim + self.text_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)
        )

    def forward(self, image, input_ids, attention_mask):
        # Image → ResNet features
        img_feat = self.cnn(image)                # [B, C, 1, 1]
        img_feat = img_feat.view(img_feat.size(0), -1)

        # Text → BERT CLS embedding
        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        txt_feat = txt_out.last_hidden_state[:, 0, :]  # CLS token

        # Fuse
        fused = torch.cat([img_feat, txt_feat], dim=1)

        # Predict
        logits = self.classifier(fused)
        return logits
