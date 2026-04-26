# HelixPR: a helix-aware periodic representation of DNA sequence for enhancer identification

This repository provides a simple two-stage pipeline for DNA sequence classification:

1. **Preprocess FASTA files into MFPE embeddings**
2. **Train a CNN classifier on the generated embeddings**

The code is organized into two scripts:

- `mfpe.py`: convert FASTA sequences into MFPE-based CSV embeddings
- `train.py`: load the generated embeddings and train a CNN classifier

---

## Overview

The preprocessing script encodes each DNA sequence with:

- **one-hot nucleotide encoding**
- **a CLS-style global composition token**
- **periodic positional encoding**

The training script then reads the saved embeddings, pads sequences to a common length, normalizes the data, and trains a lightweight 1D CNN for classification.

---

## File Structure

Recommended directory layout:

```text
project/
├── helixPR.py
├── train.py
├── data/
│   ├── raw/
│   │   ├── train/
│   │   │   ├── positive.txt
│   │   │   └── negative.txt
│   │   └── test/
│   │       ├── positive.txt
│   │       └── negative.txt
│   └── processed/
│       ├── train/
│       │   ├── positive/
│       │   └── negative/
│       └── test/
│           ├── positive/
│           └── negative/
└── best.pth
```

The preprocessing script expects FASTA-formatted `.txt` files as input.

The training script expects processed embeddings stored as CSV files under class-specific subdirectories.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install numpy pandas torch biopython scikit-learn tqdm
```

---

## Step 1: Generate Embeddings

Use `helixPR.py` to convert DNA sequences into helixPR embeddings.

### Input format

The input directory should contain `.txt` files. Each file may contain multiple sequences.

### Example

```bash
python helixPR.py \
  --input_dir /path/to/raw/train \
  --output_dir /path/to/processed/train \
  --alpha 0.05 \
  --period 10
```

Run the same process for the test set:

```bash
python helixPR.py \
  --input_dir /path/to/raw/test \
  --output_dir /path/to/processed/test \
  --alpha 0.05 \
  --period 10
```

### Output

For each input `.txt` file, the script creates a folder with the same name and saves one CSV file per sequence.


## Step 2: Train the CNN

Use `train.py` to train and evaluate the classifier.



### Example

```bash
python train.py \
  --train_dir /path/to/processed/train \
  --test_dir /path/to/processed/test \
  --batch_size 64 \
  --epochs 40 \
  --lr 1e-4 \
  --seed 42 \
  --save_path best.pth
```

---

## Arguments

### `helixPR.py`

- `--input_dir`: directory containing FASTA `.txt` files
- `--output_dir`: directory to save generated CSV embeddings
- `--alpha`: scaling factor for positional encoding
- `--period`: positional encoding period

### `train.py`

- `--train_dir`: directory containing training CSV embeddings
- `--test_dir`: directory containing test CSV embeddings
- `--batch_size`: batch size
- `--epochs`: number of training epochs
- `--lr`: learning rate
- `--seed`: random seed
- `--save_path`: path to save the trained model




