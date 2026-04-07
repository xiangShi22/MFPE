import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class SequenceCNN(nn.Module):
    def __init__(
        self,
        num_classes=2,
        input_dim=4,
        hidden_channels=(128, 256, 128, 64),
        kernel_sizes=(3, 9, 4, 4),
    ):
        super().__init__()
        h1, h2, h3, h4 = hidden_channels
        k1, k2, k3, k4 = kernel_sizes

        self.conv = nn.Sequential(
            Block(input_dim, h1, k1),
            Block(h1, h2, k2),
            Block(h2, h3, k3),
            Block(h3, h4, k4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(h4, h4 // 2),
            nn.ReLU(),
            nn.Linear(h4 // 2, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_all_data(data_dir, pad_length=None):
    data = []
    labels = []
    seq_lengths = []

    subfolders = sorted(
        folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(subfolders)

    for class_name in subfolders:
        label = label_encoder.transform([class_name])[0]
        class_dir = os.path.join(data_dir, class_name)

        for file_name in sorted(os.listdir(class_dir)):
            if not file_name.endswith(".csv"):
                continue
            sequence = pd.read_csv(os.path.join(class_dir, file_name)).values
            seq_lengths.append(len(sequence))
            data.append(sequence)
            labels.append(label)

    if not data:
        raise ValueError(f"No CSV files found in {data_dir}")

    if pad_length is None:
        pad_length = max(seq_lengths)
    else:
        pad_length = max(max(seq_lengths), pad_length)

    padded = [
        np.pad(sequence[:pad_length], ((0, pad_length - len(sequence)), (0, 0)), mode="constant")
        for sequence in data
    ]

    return np.array(padded), np.array(labels), label_encoder, pad_length


def normalize_train_test(x_train, x_test):
    flat = np.concatenate(x_train, axis=0)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test


def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    auc = roc_auc_score(labels, probs)

    return {
        "ACC": acc,
        "SN": sn,
        "SP": sp,
        "BACC": (sn + sp) / 2,
        "MCC": mcc,
        "AUC": auc,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="best.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, _, pad_length = load_all_data(args.train_dir)
    x_test, y_test, _, _ = load_all_data(args.test_dir, pad_length=pad_length)
    x_train, x_test = normalize_train_test(x_train, x_test)

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SequenceDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SequenceCNN(input_dim=x_train.shape[2], num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm.trange(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"epoch={epoch + 1} loss={loss:.6f}")

    torch.save(model.state_dict(), args.save_path)

    labels, preds, probs = evaluate(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)

    print(
        " ".join(
            [
                f"ACC={metrics['ACC']:.4f}",
                f"SN={metrics['SN']:.4f}",
                f"SP={metrics['SP']:.4f}",
                f"MCC={metrics['MCC']:.4f}",
                f"AUC={metrics['AUC']:.4f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
