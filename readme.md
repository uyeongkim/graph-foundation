# Graph Foundation Models via Pattern-Oriented Transformers

This repository provides two **independent implementations** of Transformer-based models for node classification on graph-structured data. Each model represents a distinct approach to constructing a **graph foundation model**, aiming to generalize across graph types and tasks through local pattern encoding.

---

## Included Approaches

### 1. Random Walk Transformer (RWT)

A Transformer-based model for **homogeneous graphs** that encodes node sequences obtained via random walks. It supports various walk strategies such as:

- Standard Random Walk
- Minimum-Degree Last Resort (MDLR)
- Non-Backtracking Walk (NBW)
- Node2Vec (with `p`, `q` parameters)

The model learns node representations by aggregating predictions across multiple walk sequences per node.

### 2. Hypergraph Pattern Transformer (HPT)

A pattern-based model for **hypergraphs** that extracts multi-view walk patterns for each node, including:

- Semantic paths (based on input features)
- Anonymous node paths (structure-based role encoding)
- Anonymous edge paths (hyperedge-centric role encoding)

These views are encoded using Transformers and GRUs, then aggregated for node classification.

Each model is designed and implemented **independently**, with its own dataset format, training pipeline, and model architecture.

---

## Usage

- Use the `rwt/` folder to run experiments on standard graphs with random walk tokenization and Transformer-based sequence classification.
- Use the `hpt/` folder to run experiments on hypergraph data using walk-based pattern extraction and multi-view encoders.

Each subdirectory contains:
- A `main.py` training script
- Model and encoder definitions
- Data preprocessing utilities
- Example configurations

Refer to each folder's scripts and docstrings for detailed instructions.