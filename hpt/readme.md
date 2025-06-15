# Hypergraph Pattern Transformer

A PyTorch implementation for hypergraph-based node classification using multi-view random walk patterns and Transformer models. This framework extracts semantic and anonymous patterns from hypergraphs and models them through tailored sequence encoders and attention-based aggregation.

---

## Overview

This repository introduces a pattern-based approach to node classification on hypergraphs. Instead of modeling global structure or full incidence matrices, the method performs random walks on hyperedges and nodes to extract **local hypergraph patterns**, which are then encoded and aggregated using Transformer and GRU modules.

---

## Method

### Pattern Construction
For each node:
- Perform `k` random walks of varying lengths (e.g., 2, 4, 6, 8)
- Extract:
  - **Semantic paths** (node feature sequences)
  - **Anonymous node paths** (role-based node identities)
  - **Anonymous edge paths** (role-based edge identities)

These are passed into three dedicated encoders.

### Architecture

```
Input: 
    - semantic_path          [k × L × F]
    - anonymous_node_path    [k × L × L]
    - anonymous_edge_path    [k × L × L]

Modules:
    ↓
[Semantic Encoder]   ← Transformer + PE + LayerNorm
[Anonymous Node Encoder] ← GRU + Linear
[Anonymous Edge Encoder] ← GRU + Linear

Aggregation:
    ↓
Pattern Representation ← weighted sum: semantic + wn·anonymous_node + we·anonymous_edge
    ↓
[Pattern Identifier] ← Transformer Encoder Layers
    ↓
Global Pooling (mean)
    ↓
Task Head (Linear → C classes)
```

- `wn`, `we` control the influence of anonymous paths.

---

## File Structure

```
.
├── main.py               # Training pipeline
├── model.py              # Model architecture (encoders + transformer + head)
├── data.py               # Dataset class with walk sampling and path generation
├── dataset/              # Contains preprocessed .pt files
└── temp/                 # Pattern cache (arrow format per split)
```

---

## Usage

### 1. Dataset Preparation

Preprocessed datasets must be saved in `dataset/{name}/data.pt` and `split.pt`.

Example directory:
```
dataset/
└── cora/
    ├── data.pt       # Contains edge_index, node features X, and labels Y
    └── split.pt      # 10 random train/val/test splits
```

### 2. Run Training

```bash
python main.py --target cora --num_patterns 16 --embed_dim 512 --num_heads 8 \
    --batch_size 128 --learning_rate 1e-3 --num_pattern_layers 1 --num_task_layers 1
```

**Main arguments:**
- `--target`: Dataset name (e.g., cora, pubmed, etc.)
- `--num_patterns`: Number of walk patterns per node
- `--weight_anonymous_node`, `--weight_anonymous_edge`: Relative weightings
- `--embed_dim`, `--feed_forward_dim`: Model size
- `--num_heads`: Multi-head attention heads
- `--dropout`: Dropout rate
- `--max_epoch`, `--early_stop`: Training schedule
- `--warmup`: Warmup steps for cosine scheduler

---

## Components

### `main.py`
- Entry point for training and evaluation
- Loads `.pt` graph and split files
- Generates or loads walk patterns
- Trains model over 10 random splits with early stopping

### `data.py`
- `HypergraphPatternDataset`:
  - Generates `k` node/edge walks per node
  - Constructs:
    - `semantic_path` (feature vector sequences)
    - `anonymous_node_path` (L×L one-hot role matrix)
    - `anonymous_edge_path` (L×L one-hot role matrix)

### `model.py`
- `HypergraphPatternMachine`: Combines three encoders and pattern aggregation
- `SequentialEncoderTF`: Transformer encoder with sinusoidal positional encoding
- `SequentialEncoderGRU`: GRU encoder for anonymous path
- Final prediction via `task_head` over mean pooled pattern embedding

---

## Output

- Evaluates accuracy over 10 random splits
- Logs average accuracy and standard deviation to `result.txt`

Example:
```
cora (k=16, wn=1, we=1, d=512, ...) : accuracy = 87.43, std = 0.58
```

---

## Requirements

- Python 3.8+
- PyTorch >= 1.12
- Huggingface `datasets`
- `transformers`
- `numpy`, `scipy`
- Optional: `tqdm`, `pickle`

---
