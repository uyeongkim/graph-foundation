# Random Walk Transformer for Graph Node Classification

A PyTorch implementation of Transformer-based graph node classification using random walk sequences. This repository provides a comprehensive framework for training and evaluating Transformer models on graph-structured data through various random walk strategies.

## Overview

This implementation explores the application of Transformer architectures to graph node classification by converting graph structure into sequential data through random walks. The approach leverages the sequential modeling capabilities of Transformers while preserving graph topology through carefully designed walk strategies.

## Method

### Random Walk Generation

The framework supports four distinct random walk strategies:

1. **Standard Random Walk**: Uniform transition probabilities proportional to node degrees
2. **Minimum Degree Last Resort (MDLR)**: Biased transitions favoring lower-degree neighbors  
3. **Non-Backtracking Walk (NBW)**: Eliminates immediate return transitions to previous nodes
4. **Node2Vec**: Parameterized walks with return parameter `p` and in-out parameter `q`

### Model Architecture

```
Input: Tokenized random walk sequences
    ↓
Token Embedding (vocab_size → d_model)
    ↓
Positional Encoding
    ↓
Transformer Encoder Layers
    ↓
Sequence Aggregation {mean, max, first, last, cls}
    ↓
Classification Head (d_model → num_classes)
```

The model supports optional contrastive learning through an auxiliary embedding output that enforces similarity between walks from the same node.

## Installation

```bash
git clone <repository-url>
cd random-walk-transformer
pip install -r requirements.txt
```

### Optional Dependencies

For improved attention computation performance:
```bash
pip install flash-attn --no-build-isolation
```

## Usage

### Dataset Preparation

Convert graph datasets to optimized local format:

```bash
python create_dataset.py \
    --dataset {cora,pubmed,amazon-ratings,roman-empire} \
    --walk_type {standard,mdlr,nbw,node2vec} \
    --output_path <path> \
    [--p P_VALUE] [--q Q_VALUE]
```

**Parameters:**
- `--dataset`: Target graph dataset
- `--walk_type`: Random walk strategy
- `--repetitions`: Number of walk sequences per node (default: 100)
- `--walk_length`: Total sequence length (default: 300)
- `--p`, `--q`: Node2Vec parameters for biased walks

### Model Training

```bash
python main.py \
    --data_path <dataset_path> \
    --d_model <embedding_dim> \
    --n_layers <num_layers> \
    --n_heads <num_heads> \
    --lr <learning_rate> \
    --batch_size <batch_size> \
    --num_epochs <epochs>
```

**Key Parameters:**
- `--d_model`: Transformer hidden dimension
- `--n_layers`: Number of encoder layers  
- `--n_heads`: Multi-head attention heads
- `--lambda`: Contrastive loss weight (0.0 disables contrastive learning)

## Configuration

The framework uses a three-tier configuration system:

### DataConfig
- Dataset path and preprocessing parameters
- Random walk configuration (length, repetitions, p/q values)
- Tokenization settings

### ModelConfig  
- Transformer architecture parameters
- Training hyperparameters (learning rate, batch size, epochs)
- Contrastive learning settings

### SystemConfig
- Hardware and system settings
- Logging and checkpointing configuration
- Experiment tracking parameters

## Supported Datasets

| Dataset | Nodes | Edges | Classes | Type | Description |
|---------|-------|-------|---------|------|-------------|
| **Cora** | 2,708 | 5,429 | 7 | Citation | Computer science papers with citation links |
| **PubMed** | 19,717 | 44,338 | 3 | Citation | Biomedical papers from PubMed database |
| **CiteSeer** | 3,327 | 4,732 | 6 | Citation | Computer science papers and citations |
| **Amazon-Ratings** | 24,492 | 93,050 | 5 | Heterophilous | Product co-purchasing network with ratings |
| **Roman-Empire** | 22,662 | 32,927 | 18 | Heterophilous | Historical network of Roman empire entities |

### Dataset Characteristics

| Property | Citation Networks | Heterophilous Networks |
|----------|------------------|------------------------|
| **Homophily** | High | Low |
| **Node Features** | Bag-of-words | Categorical/numerical |
| **Edge Semantics** | Citation relationships | Co-occurrence/interaction |
| **Class Distribution** | Balanced | Imbalanced |

## Experimental Setup

### Training Protocol
1. Generate random walk sequences for each node
2. Tokenize sequences using WordPiece or custom tokenizer
3. Train Transformer with cross-entropy loss
4. Optional: Add contrastive loss for embedding regularization
5. Evaluate using majority voting across multiple walks per node

### Evaluation Metrics
- **Node-level Accuracy**: Primary metric using majority voting
- **Walk-level Accuracy**: Individual walk classification performance
- **Training Metrics**: Loss curves, learning rate schedules

## Implementation Details

### Automatic Checkpoint Resuming
The system automatically resumes training from checkpoints when the same configuration is used:

```bash
# Initial training
python main.py --data_path data/cora --lr 1e-5 --d_model 128

# Automatic resume on re-run
python main.py --data_path data/cora --lr 1e-5 --d_model 128
```

Checkpoint directories are determined by configuration hash, ensuring consistent resuming behavior.

### Memory and Performance Optimizations
- **Mixed Precision Training**: Automatic mixed precision (AMP) support
- **FlashAttention**: Optional integration for memory-efficient attention
- **Local Data Format**: Arrow/Parquet format for fast dataset loading
- **Gradient Accumulation**: Support for effective large batch training

### Majority Voting Inference
Test-time prediction aggregates multiple random walks per node:

```python
for node in test_nodes:
    walks = generate_walks(node, num_walks=repetitions)
    predictions = [model(walk) for walk in walks]
    final_prediction = majority_vote(predictions)
```

## File Structure

```
new_src/
├── main.py              # Main training script
├── config.py            # Configuration management
├── trainer.py           # Training loop implementation
├── model.py             # Transformer classifier
├── create_dataset.py    # Dataset preparation utility
├── logger.py            # Logging utilities
└── data/
    └── module.py        # Data loading and preprocessing
```

## Examples

### Basic Training
```bash
# Prepare Cora dataset
python create_dataset.py --dataset cora --walk_type standard --output_path data/cora

# Train model
python main.py --data_path data/cora
```

### Contrastive Learning
```bash
# Enable contrastive loss
python main.py --data_path data/cora --lambda 0.1 --max_neg 5
```

### Node2Vec Walks
```bash
# Generate Node2Vec walks
python create_dataset.py --dataset pubmed --walk_type node2vec --p 2.0 --q 0.5 --output_path data/pubmed_node2vec

# Train on Node2Vec walks
python main.py --data_path data/pubmed_node2vec --d_model 256 --n_layers 8
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers
- Datasets (HuggingFace)
- NetworkX
- SciPy
- NumPy
- Rich (for console output)
- Weights & Biases (optional, for experiment tracking)

## License

MIT License - see LICENSE file for details.