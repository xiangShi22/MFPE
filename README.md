# DNA Sequence Classification with MFPE + CNN

This repository provides a simple two-stage pipeline for DNA sequence classification:

1. **Preprocess FASTA files into MFPE embeddings**
2. **Train a CNN classifier on the generated embeddings**

The code is organized into two scripts:

- `mfpe.py`: convert FASTA sequences into MFPE-based CSV embeddings
- `cnn.py`: load the generated embeddings and train a CNN classifier

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
├── open_source_cleaned_code.py
├── open_source_cleaned_train.py
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

Use `open_source_cleaned_code.py` to convert FASTA sequences into CSV embeddings.

### Input format

The input directory should contain FASTA `.txt` files. Each file may contain multiple sequences.

### Example

```bash
python open_source_cleaned_code.py \
  --input_dir /path/to/raw/train \
  --output_dir /path/to/processed/train \
  --alpha 0.05 \
  --period 10
```

Run the same process for the test set:

```bash
python open_source_cleaned_code.py \
  --input_dir /path/to/raw/test \
  --output_dir /path/to/processed/test \
  --alpha 0.05 \
  --period 10
```

### Output

For each input `.txt` file, the script creates a folder with the same name and saves one CSV file per sequence.

Each CSV has 4 columns:

- `A`
- `C`
- `G`
- `T`

Each row corresponds to either:

- the prepended CLS-style token, or
- an encoded nucleotide position

---

## Step 2: Train the CNN

Use `open_source_cleaned_train.py` to train and evaluate the classifier.



### Example

```bash
python open_source_cleaned_train.py \
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

### `mfpe.py`

- `--input_dir`: directory containing FASTA `.txt` files
- `--output_dir`: directory to save generated CSV embeddings
- `--alpha`: scaling factor for positional encoding
- `--period`: positional encoding period

### `cnn.py`

- `--train_dir`: directory containing training CSV embeddings
- `--test_dir`: directory containing test CSV embeddings
- `--batch_size`: batch size
- `--epochs`: number of training epochs
- `--lr`: learning rate
- `--seed`: random seed
- `--save_path`: path to save the trained model

---

## Notes

- Sequences containing `N` are skipped during preprocessing.
- Sequence embeddings are padded to the maximum length found in the dataset.
- Normalization statistics are computed from the training set and applied to both training and test data.
- The current training script is designed for standard classification experiments and saves the final model weights to `best.pth`.

---

## Example Workflow

```bash
python open_source_cleaned_code.py \
  --input_dir ./data/raw/train \
  --output_dir ./data/processed/train \
  --alpha 0.05 \
  --period 10

python open_source_cleaned_code.py \
  --input_dir ./data/raw/test \
  --output_dir ./data/processed/test \
  --alpha 0.05 \
  --period 10

python open_source_cleaned_train.py \
  --train_dir ./data/processed/train \
  --test_dir ./data/processed/test \
  --batch_size 64 \
  --epochs 40 \
  --lr 1e-4 \
  --seed 42 \
  --save_path best.pth
```

---

