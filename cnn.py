import os
import tqdm
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

class SequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        return self.conv(x)


class SequenceCNN(nn.Module):
    def __init__(self,
                 num_classes=2,
                 input_dim=4,
                 hidden_channels=(128, 256, 128, 64),
                 kernel_sizes=(3,9,4,4)):

        super().__init__()
        h1, h2, h3, h4 = hidden_channels
        k1, k2, k3, k4 = kernel_sizes

        self.conv = nn.Sequential(
            Block(input_dim, h1, k1),
            Block(h1, h2, k2),
            Block(h2, h3, k3),
            Block(h3, h4, k4)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(h4, h4 // 2),
            nn.ReLU(),
            nn.Linear(h4 // 2, num_classes)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def load_all_data(data_dir, pad_length = None):
    data, labels, seq_lengths = [], [], []

    subfolders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    label_encoder = LabelEncoder()
    label_encoder.fit(subfolders)

    for class_name in subfolders:
        label = label_encoder.transform([class_name])[0]
        class_dir = os.path.join(data_dir, class_name)

        for file in sorted(list(os.listdir(class_dir))):
            if file.endswith(".csv"):
                seq = pd.read_csv(os.path.join(class_dir, file)).values
                # seq = pd.read_csv(os.path.join(class_dir, file)).values.T
                seq_lengths.append(len(seq))
                data.append(seq)
                labels.append(label)

    if pad_length is None:
        pad_length = max(seq_lengths)
    else:
        pad_length = max(max(seq_lengths), pad_length)

    padded = [
        np.pad(seq[:pad_length], ((0, pad_length-len(seq)), (0,0)), mode="constant")
        for seq in data
    ]

    return np.array(padded), np.array(labels), label_encoder, pad_length


def normalize_train_val_test(X_train, X_test):
    flat = np.concatenate(X_train, axis=0)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test  = (X_test - mean) / std
    return X_train, X_test


def evaluate_preds(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)                # shape (B, 2)
            probs = torch.softmax(logits, dim=1)[:, 1]  
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def compute_metrics(labels, preds, probs=None):
    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Sn  = TP / (TP + FN) if (TP + FN) > 0 else 0
    Sp  = TN / (TN + FP) if (TN + FP) > 0 else 0
    Bacc = (Sn + Sp) / 2

    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = (TP * TN - FP * FN) / denom if denom > 0 else 0

    auc = None
    if probs is not None:
        auc = roc_auc_score(labels, probs)

    return acc, Sn, Sp, MCC, auc

def extract_features(model, dataloader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x = x.permute(0,2,1)
            x = model.conv(x)
            x = model.pool(x).squeeze(-1)  
            features.append(x.cpu())
            labels.append(y)
    return torch.cat(features, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_full, y_train_full, label_encoder, pad_length = load_all_data("/data/train")
    X_train, y_train = X_train_full, y_train_full
    X_test, y_test, _, _ = load_all_data("/data/test")

    X_train, X_test = normalize_train_val_test(X_train, X_test)

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(SequenceDataset(X_test, y_test), batch_size=64)

    model = SequenceCNN(input_dim=X_train.shape[2], num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in tqdm.tqdm(range(40)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "best.pth")

    test_labels, test_preds, test_probs = evaluate_preds(model, test_loader, device)
    acc, Sn, Sp, MCC, auc = compute_metrics(test_labels, test_preds, test_probs)

    print(f"ACC={acc:.4f} SN={Sn:.4f} SP={Sp:.4f} MCC={MCC:.4f} auc={auc:.4f}")

if __name__ == "__main__":
    main()
