#!/usr/bin/env python3
"""
Local Dataset Creator for Random Walk Transformer.

Creates Arrow/Parquet format datasets locally with beautiful rich console output.
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
from datasets import Dataset, DatasetDict, ClassLabel
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes, Compose
from torch_geometric.utils import to_networkx

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.live import Live
from rich.layout import Layout

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from logger import setup_logging, get_logger

# Global variables for multiprocessing
SAMPLER: "RandomWalkSampler" = None
console = Console()
logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    dataset: str
    walk_type: str
    output_path: str
    num_restarts: int = 1
    repetitions: int = 100
    p: float = 1.0
    q: float = 1.0
    mask: bool = False
    cycle_aware: bool = False
    anonymize: bool = False
    walk_length: int = 300
    
    def __post_init__(self):
        """Validate configuration."""
        if self.walk_type not in ["standard", "mdlr", "nbw", "node2vec"]:
            raise ValueError(f"Invalid walk_type: {self.walk_type}")
        if self.dataset.lower() not in ["cora", "pubmed", "amazon-ratings", "roman-empire"]:
            raise ValueError(f"Unsupported dataset: {self.dataset}")


class RandomWalkSampler:
    """Optimized random walk sampler with torch tensors."""
    
    def __init__(self, transitions, graph, wtype, walk_len, group_size, repetitions):
        self.graph = graph
        self.wtype = wtype
        self.walk_len = walk_len
        self.group_size = group_size
        self.repetitions = repetitions
        
        FIRST = {"standard", "mdlr"}
        SECOND = {"nbw", "node2vec"}
        
        if wtype in FIRST:
            # First-order transitions: node-to-node
            P_csr = transitions.tocsr()
            self.NEIGHBORS, self.CUMPROBS = [], []
            for i in range(P_csr.shape[0]):
                row = P_csr.getrow(i)
                nbrs = torch.tensor(row.indices, dtype=torch.long)
                cum = torch.tensor(row.data, dtype=torch.float).cumsum(0)
                self.NEIGHBORS.append(nbrs)
                self.CUMPROBS.append(cum)
        
        elif wtype in SECOND:
            # Second-order transitions: edge-to-edge
            P, src, dst = transitions
            self.src = torch.tensor(src, dtype=torch.long)
            self.dst = torch.tensor(dst, dtype=torch.long)
            M = P.shape[0]
            
            # Build edge-to-edge tables
            P_csr = P.tocsr()
            self.NEIGHBORS2, self.CUMPROBS2 = [], []
            for eid in range(M):
                row = P_csr.getrow(eid)
                nbrs = torch.tensor(row.indices, dtype=torch.long)
                cum = torch.tensor(row.data, dtype=torch.float).cumsum(0)
                self.NEIGHBORS2.append(nbrs)
                self.CUMPROBS2.append(cum)
            
            # Initial edge lists per node
            init_edges = [[] for _ in range(graph.number_of_nodes())]
            for eid, u in enumerate(src):
                init_edges[u].append(eid)
            
            self.NEIGHBORS0, self.CUMPROBS0 = [], []
            for lst in init_edges:
                if lst:
                    t = torch.tensor(lst, dtype=torch.long)
                    cum = torch.ones(len(lst), dtype=torch.float).cumsum(0)
                else:
                    t, cum = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float)
                self.NEIGHBORS0.append(t)
                self.CUMPROBS0.append(cum)
    
    @staticmethod
    def _draw(cum: torch.Tensor) -> int:
        """Draw random index from cumulative probabilities."""
        r = torch.rand(1).item()
        return torch.searchsorted(cum, r, right=False).item()
    
    def walk_first_order(self, start: int) -> List[int]:
        """Generate first-order random walk."""
        walk, cur = [start], start
        for _ in range(self.walk_len - 1):
            nbrs, cum = self.NEIGHBORS[cur], self.CUMPROBS[cur]
            if nbrs.numel() == 0:
                break
            cur = int(nbrs[self._draw(cum)])
            walk.append(cur)
        return walk
    
    def walk_second_order(self, start: int) -> List[int]:
        """Generate second-order random walk."""
        walk = [start]
        nbrs0, cum0 = self.NEIGHBORS0[start], self.CUMPROBS0[start]
        if nbrs0.numel() == 0:
            return walk
        
        eid = int(nbrs0[self._draw(cum0)])
        v = int(self.dst[eid].item())
        walk.append(v)
        
        for _ in range(self.walk_len - 2):
            nbrs2, cum2 = self.NEIGHBORS2[eid], self.CUMPROBS2[eid]
            if nbrs2.numel() == 0:
                break
            eid = int(nbrs2[self._draw(cum2)])
            v = int(self.dst[eid].item())
            walk.append(v)
        
        return walk
    
    def records_for_node(self, arg: Tuple[int, int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate records for a single node."""
        node, label, record_args = arg
        records = []
        
        for rep in range(self.repetitions):
            walks = []
            for _ in range(self.group_size):
                if self.wtype in {"standard", "mdlr"}:
                    walk = self.walk_first_order(node)
                else:
                    walk = self.walk_second_order(node)
                walks.append(self._walk_to_string(walk, record_args))
            
            records.append({
                "string_rw": "[CLS] " + " [SEP] ".join(walks),
                "label": label,
                "group_idx": rep,
                "node_idx": node,
            })
        
        return records
    
    def _walk_to_string(self, walk: List[int], args: Dict[str, Any]) -> str:
        """Convert walk to string representation."""
        if args.get("cycle_aware", False):
            chunked = self._chunk_by_cycles(walk)
            sub_args = args.copy()
            sub_args["cycle_aware"] = False
            parts = [self._walk_to_string(seg, sub_args).replace(" ", "-") for seg in chunked]
            return " ".join(parts)
        
        if args.get("anonymize", False):
            idx_map = {}
            anonymized = []
            for n in walk:
                if n not in idx_map:
                    idx_map[n] = len(idx_map)
                anonymized.append(idx_map[n])
            return " ".join(map(str, anonymized))
        
        if args.get("mask", False):
            target = walk[0]
            return " ".join("[MASK]" if x == target else str(x) for x in walk)
        
        return " ".join(map(str, walk))
    
    def _chunk_by_cycles(self, walk: List[int]) -> List[List[int]]:
        """Split walk by detected cycles."""
        first_last: Dict[int, List[int]] = {}
        for i, n in enumerate(walk):
            if n in first_last:
                first_last[n][1] = i
            else:
                first_last[n] = [i, i]
        
        cycles = [(s, e) for s, e in first_last.values() if e > s]
        if not cycles:
            return [walk]
        
        start, end = max(cycles, key=lambda x: x[1] - x[0])
        segments: List[List[int]] = []
        if start > 0:
            segments += self._chunk_by_cycles(walk[:start])
        segments.append(walk[start:end + 1])
        if end + 1 < len(walk):
            segments += self._chunk_by_cycles(walk[end + 1:])
        return segments


class TransitionBuilder:
    """Build transition matrices for different walk types."""
    
    @staticmethod
    def build_transitions(
        graph: nx.Graph, 
        walk_type: str, 
        p: float = 1.0, 
        q: float = 1.0
    ):
        """Build transition matrix based on walk type."""
        A = nx.to_scipy_sparse_array(graph, format="csr", dtype=float)
        
        if walk_type == "standard":
            return TransitionBuilder._simple_transition(A)
        elif walk_type == "mdlr":
            return TransitionBuilder._mdlr_transition(A)
        elif walk_type == "nbw":
            return TransitionBuilder._nbw_transition(A)
        elif walk_type == "node2vec":
            return TransitionBuilder._node2vec_transition(A, p, q)
        else:
            raise ValueError(f"Unknown walk type: {walk_type}")
    
    @staticmethod
    def _simple_transition(A: sp.csr_matrix) -> sp.csr_matrix:
        """Standard random walk transition."""
        deg = np.diff(A.indptr)
        inv_deg = np.where(deg == 0, 0.0, 1.0 / deg)
        return sp.diags(inv_deg).dot(A).tocsr()
    
    @staticmethod
    def _mdlr_transition(A: sp.csr_matrix) -> sp.csr_matrix:
        """Minimum degree last resort transition."""
        indptr, indices = A.indptr, A.indices
        deg = np.diff(indptr)
        rows = np.repeat(np.arange(A.shape[0]), deg)
        cols = indices
        min_deg = np.minimum(deg[rows], deg[cols])
        weights = 1.0 / np.maximum(min_deg, 1.0)
        W = sp.coo_matrix((weights, (rows, cols)), shape=A.shape)
        row_sum = np.asarray(W.sum(axis=1)).flatten()
        inv_row = np.where(row_sum == 0, 0.0, 1.0 / row_sum)
        return sp.diags(inv_row).dot(W).tocsr()
    
    @staticmethod
    def _nbw_transition(A: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Non-backtracking walk transition."""
        src = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
        dst = A.indices
        N, M = A.shape[0], src.size
        
        out_deg = np.diff(A.indptr)
        denom = out_deg[dst] - 1
        
        # Build edge-to-edge transitions
        dest_inc = sp.csr_matrix((np.ones(M), (np.arange(M), dst)), shape=(M, N))
        src_inc = sp.csr_matrix((np.ones(M), (src, np.arange(M))), shape=(N, M))
        C = (dest_inc @ src_inc).tocoo()
        
        # Detect back-edges
        key = src.astype(np.int64) * N + dst
        rev_key = dst.astype(np.int64) * N + src
        order = np.argsort(key)
        sorted_k = key[order]
        pos = np.searchsorted(sorted_k, rev_key)
        rev_edge = np.full(M, -1, np.int64)
        m = (pos < M) & (sorted_k[pos] == rev_key)
        rev_edge[m] = order[pos[m]]
        
        mask = C.col != rev_edge[C.row]
        rows, cols = C.row[mask], C.col[mask]
        data = 1.0 / denom[rows]
        
        P = sp.csr_matrix((data, (rows, cols)), shape=(M, M))
        return P, src, dst
    
    @staticmethod
    def _node2vec_transition(A: sp.csr_matrix, p: float, q: float) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Node2vec transition with p, q parameters."""
        N = A.shape[0]
        src = np.repeat(np.arange(N), np.diff(A.indptr)).astype(np.int64)
        dst = A.indices.astype(np.int64)
        M = src.size
        
        dest_inc = sp.csr_matrix((np.ones(M), (np.arange(M), dst)), shape=(M, N))
        src_inc = sp.csr_matrix((np.ones(M), (src, np.arange(M))), shape=(N, M))
        C = (dest_inc @ src_inc).tocoo()
        
        rows, cols = C.row, C.col
        u = src[rows]
        v = dst[rows]
        w = dst[cols]
        
        edge_key = (src * N + dst).astype(np.int64)
        order = np.argsort(edge_key)
        sorted_k = edge_key[order]
        
        key_uw = (u * N + w).astype(np.int64)
        pos = np.searchsorted(sorted_k, key_uw)
        mask = pos < M
        is_edge = np.empty_like(key_uw, dtype=bool)
        is_edge[mask] = sorted_k[pos[mask]] == key_uw[mask]
        is_edge[~mask] = False
        
        unnorm = np.where(
            w == u, 1.0 / p,
            np.where(is_edge, 1.0, 1.0 / q)
        )
        row_sum = np.bincount(rows, weights=unnorm, minlength=M)
        probs = unnorm / row_sum[rows]
        
        P = sp.csr_matrix((probs, (rows, cols)), shape=(M, M))
        return P, src, dst


class DatasetCreator:
    """Main dataset creation class with rich console output."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.console = Console()
        self.logger = get_logger()
    
    def create_dataset(self):
        """Main dataset creation pipeline."""
        self.logger.print_banner(
            "🎯 Dataset Creation Started",
            f"Creating {self.config.dataset} dataset with {self.config.walk_type} walks"
        )
        
        # Load graph data
        graph, labels, data = self._load_graph_data()
        
        # Build transitions
        transitions = self._build_transitions(graph)
        
        # Generate random walks
        records = self._generate_walks(graph, labels, transitions)
        
        # Create dataset splits
        dataset_dict = self._create_splits(records, data)
        
        # Save dataset
        self._save_dataset(dataset_dict)
        
        self.logger.print_banner("🎉 Dataset Creation Complete!", f"Saved to: {self.config.output_path}")
    
    def _load_graph_data(self):
        """Load and preprocess graph data."""
        self.logger.info(f"📊 Loading {self.config.dataset} dataset...")
        
        transform = Compose([ToUndirected(), RemoveIsolatedNodes()])
        
        if self.config.dataset.lower() in ["cora", "pubmed"]:
            data = Planetoid(root="data/raw", name=self.config.dataset, transform=transform)[0]
        elif self.config.dataset.lower() in ["amazon-ratings", "roman-empire"]:
            data = HeterophilousGraphDataset(
                root="data/raw", 
                name=self.config.dataset, 
                pre_transform=transform
            )[0]
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
        
        graph = to_networkx(data, to_undirected=True)  # Force undirected
        labels = data.y.numpy().astype(np.int64)
        
        # Log graph statistics
        graph_stats = {
            "Nodes": graph.number_of_nodes(),
            "Edges": graph.number_of_edges(),
            "Classes": len(np.unique(labels)),
            "Average Degree": f"{2 * graph.number_of_edges() / graph.number_of_nodes():.2f}",
            "Is Connected": nx.is_connected(graph)
        }
        
        self.logger.print_config_table(graph_stats, "📈 Graph Statistics")
        return graph, labels, data
    
    def _build_transitions(self, graph):
        """Build transition matrices."""
        self.logger.info(f"🔧 Building {self.config.walk_type} transition matrix...")
        
        transitions = TransitionBuilder.build_transitions(
            graph, self.config.walk_type, self.config.p, self.config.q
        )
        
        self.logger.success(f"✅ Transition matrix built")
        return transitions
    
    def _generate_walks(self, graph, labels, transitions):
        """Generate random walks using multiprocessing."""
        walk_len = self.config.walk_length // self.config.num_restarts
        group_size = self.config.num_restarts
        
        self.logger.info(f"🚶 Generating random walks...")
        walk_config = {
            "Walk Length": walk_len,
            "Group Size": group_size,
            "Repetitions": self.config.repetitions,
            "Total Walks per Node": group_size * self.config.repetitions,
            "Workers": int(cpu_count() * 0.8)
        }
        self.logger.print_config_table(walk_config, "🚶 Walk Configuration")
        
        # Prepare arguments for multiprocessing
        init_args = (transitions, graph, self.config.walk_type, walk_len, group_size, self.config.repetitions)
        func_args = [
            (n, int(labels[n]), {
                "mask": self.config.mask,
                "cycle_aware": self.config.cycle_aware,
                "anonymize": self.config.anonymize
            }) 
            for n in range(graph.number_of_nodes())
        ]
        
        # Generate walks with progress bar
        records = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            
            task = progress.add_task("Generating walks...", total=len(func_args))
            
            with Pool(
                int(cpu_count() * 0.8), 
                initializer=_init_worker, 
                initargs=init_args
            ) as pool:
                for chunk in pool.imap_unordered(_worker_records_for_node, func_args, chunksize=20):
                    records.extend(chunk)
                    progress.update(task, advance=1)
        
        self.logger.success(f"✅ Generated {len(records):,} walk records")
        return records
    
    def _create_splits(self, records, data):
        """Create train/validation/test splits."""
        self.logger.info("📊 Creating dataset splits...")
        
        # Convert to HuggingFace dataset
        master_dataset = self._records_to_dataset(records)
        
        if self.config.dataset.lower() in ["cora", "pubmed"]:
            # Single split for Cora/PubMed
            train_idx = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            val_idx = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            test_idx = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            
            node_idxs = np.array(master_dataset["node_idx"])
            train_mask = np.isin(node_idxs, train_idx)
            val_mask = np.isin(node_idxs, val_idx)
            test_mask = np.isin(node_idxs, test_idx)
            
            # Filter to only include nodes in splits
            valid_mask = train_mask | val_mask | test_mask
            master_dataset = master_dataset.select(np.where(valid_mask)[0])
            
            # Create DatasetDict
            dataset_dict = DatasetDict({
                "train": master_dataset.filter(lambda x, idx: train_mask[valid_mask][idx], with_indices=True),
                "validation": master_dataset.filter(lambda x, idx: val_mask[valid_mask][idx], with_indices=True),
                "test": master_dataset.filter(lambda x, idx: test_mask[valid_mask][idx], with_indices=True)
            })
            
        else:
            # Multiple splits for heterophilous datasets
            # For now, use the first split
            train_idx = data.train_mask[:, 0].nonzero(as_tuple=True)[0].cpu().numpy()
            val_idx = data.val_mask[:, 0].nonzero(as_tuple=True)[0].cpu().numpy()
            test_idx = data.test_mask[:, 0].nonzero(as_tuple=True)[0].cpu().numpy()
            
            node_idxs = np.array(master_dataset["node_idx"])
            train_mask = np.isin(node_idxs, train_idx)
            val_mask = np.isin(node_idxs, val_idx)
            test_mask = np.isin(node_idxs, test_idx)
            
            # Filter and create splits
            valid_mask = train_mask | val_mask | test_mask
            master_dataset = master_dataset.select(np.where(valid_mask)[0])
            
            dataset_dict = DatasetDict({
                "train": master_dataset.filter(lambda x, idx: train_mask[valid_mask][idx], with_indices=True),
                "validation": master_dataset.filter(lambda x, idx: val_mask[valid_mask][idx], with_indices=True),
                "test": master_dataset.filter(lambda x, idx: test_mask[valid_mask][idx], with_indices=True)
            })
        
        # Log split statistics
        split_stats = {}
        for split_name, split_data in dataset_dict.items():
            split_stats[f"{split_name.title()} Split"] = f"{len(split_data):,} samples"
        
        self.logger.print_config_table(split_stats, "📊 Dataset Splits")
        return dataset_dict
    
    def _records_to_dataset(self, records):
        """Convert records to HuggingFace Dataset."""
        dataset = Dataset.from_dict({
            key: [record[key] for record in records] 
            for key in records[0].keys()
        })
        
        # Cast label column
        unique_labels = sorted(set(dataset["label"]))
        dataset = dataset.cast_column(
            "label", 
            ClassLabel(num_classes=len(unique_labels), names=[str(label) for label in unique_labels])
        )
        
        return dataset
    
    def _save_dataset(self, dataset_dict):
        """Save dataset to disk in Arrow format."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"💾 Saving dataset to {output_path}...")
        
        with self.console.status("[bold green]Saving dataset..."):
            dataset_dict.save_to_disk(str(output_path))
        
        # Log saved files
        saved_files = list(output_path.rglob("*"))
        total_size = sum(f.stat().st_size for f in saved_files if f.is_file()) / (1024 * 1024)  # MB
        
        save_stats = {
            "Output Path": str(output_path),
            "Total Files": len([f for f in saved_files if f.is_file()]),
            "Total Size": f"{total_size:.2f} MB",
            "Format": "Arrow (Parquet)"
        }
        
        self.logger.print_config_table(save_stats, "💾 Saved Dataset Info")


# Multiprocessing worker functions
def _init_worker(transitions, graph, wtype, walk_len, group_size, repetitions):
    """Initialize worker process."""
    global SAMPLER
    SAMPLER = RandomWalkSampler(transitions, graph, wtype, walk_len, group_size, repetitions)


def _worker_records_for_node(arg):
    """Worker function for generating records."""
    return SAMPLER.records_for_node(arg)


def main():
    """Main function with argument parsing."""
    setup_logging(log_dir="logs", enable_file_logging=True)
    
    parser = argparse.ArgumentParser(
        description="Create local dataset for Random Walk Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument("--dataset", required=True, 
                       choices=["cora", "pubmed", "amazon-ratings", "roman-empire"],
                       help="Dataset name")
    parser.add_argument("--walk_type", default="standard", 
                       choices=["standard", "mdlr", "nbw", "node2vec"],
                       help="Random walk type")
    parser.add_argument("--output_path", required=True,
                       help="Output directory path")
    
    # Walk parameters
    parser.add_argument("--num_restarts", type=int, default=1,
                       help="Number of restart groups")
    parser.add_argument("--repetitions", type=int, default=100,
                       help="Repetitions per node")
    parser.add_argument("--walk_length", type=int, default=300,
                       help="Total walk length")
    parser.add_argument("--p", type=float, default=1.0,
                       help="Return parameter (for node2vec)")
    parser.add_argument("--q", type=float, default=1.0,
                       help="In-out parameter (for node2vec)")
    
    # Processing options
    parser.add_argument("--mask", action="store_true",
                       help="Use [MASK] token for target node")
    parser.add_argument("--cycle_aware", action="store_true",
                       help="Enable cycle-aware processing")
    parser.add_argument("--anonymize", action="store_true",
                       help="Anonymize node IDs")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DatasetConfig(
        dataset=args.dataset,
        walk_type=args.walk_type,
        output_path=args.output_path,
        num_restarts=args.num_restarts,
        repetitions=args.repetitions,
        p=args.p,
        q=args.q,
        mask=args.mask,
        cycle_aware=args.cycle_aware,
        anonymize=args.anonymize,
        walk_length=args.walk_length
    )
    
    # Create dataset
    creator = DatasetCreator(config)
    creator.create_dataset()


if __name__ == "__main__":
    main()