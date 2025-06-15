import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


class HypergraphPatternDataset(Dataset):
    def __init__(self, input_nodes, node_features, node_labels, node2edge, edge2node, k, walk_lengths):
        self.input_nodes = input_nodes
        self.X = node_features
        self.Y = node_labels
        self.n2e = node2edge
        self.e2n = edge2node
        self.k = k
        self.walk_lengths = walk_lengths
        
        self.precomputed_node_walks = []
        self.precomputed_edge_walks = []
        for n in self.input_nodes:
            cur_node_walk = []
            cur_edge_walk = []
            for _ in range(self.k):
                node_walk, edge_walk = self.random_walk(n, random.choice(self.walk_lengths))
                cur_node_walk.append(node_walk)
                cur_edge_walk.append(edge_walk)
            self.precomputed_node_walks.append(cur_node_walk)
            self.precomputed_edge_walks.append(cur_edge_walk)
        
    def random_walk(self, start_node, walk_length):
        node_walk = [start_node]
        edge_walk = []
        for _ in range(walk_length):
            last_node = node_walk[-1]
            last_edge = np.random.choice(self.n2e[last_node])
            edge_walk.append(last_edge)
            next_node = np.random.choice(self.e2n[last_edge])
            node_walk.append(next_node)
        node_walk.pop()
            
        return node_walk, edge_walk
    
    def generate_semantic_path(self, node_walk, max_length):
        # semantic path with node features
        semantic_path = []
        for w in node_walk:
            node_idx = w
            node_feat = self.X[node_idx]
            semantic_path.append(node_feat)
        semantic_path = torch.stack(semantic_path, dim=0)
        # padding
        semantic_padding = torch.tensor([[-1e9 for _ in range(self.X.shape[1])] for _ in range(max_length - len(node_walk))])
        semantic_path = torch.cat([semantic_path, semantic_padding], dim=0)
        
        return semantic_path
    
    def generate_anonymous_path(self, walk, max_length):
        # anonymous path
        w2idx = {}
        for w in walk:
            if w not in w2idx:
                w2idx[w] = len(w2idx)
        anonymous_walk = np.array([w2idx[w] for w in walk])
        anonymous_path = torch.zeros(len(anonymous_walk), max_length)
        for i, idx in enumerate(anonymous_walk):
            anonymous_path[i, np.where(anonymous_walk==idx)[0]] = 1   
        # padding
        anonymous_padding = torch.tensor([[-1e9 for _ in range(max_length)] for _ in range(max_length - len(walk))])
        anonymous_path = torch.cat([anonymous_path, anonymous_padding], dim=0)
        
        return anonymous_path
    
    def __len__(self):
        return len(self.input_nodes)

    def __getitem__(self, idx):
        start_node = self.input_nodes[idx]
        node_walks = self.precomputed_node_walks[idx]
        edge_walks = self.precomputed_edge_walks[idx]
        semantic_path = [self.generate_semantic_path(nw, max(self.walk_lengths)) for nw in node_walks]
        anonymous_node_path = [self.generate_anonymous_path(nw, max(self.walk_lengths)) for nw in node_walks]
        anonymous_edge_path = [self.generate_anonymous_path(ew, max(self.walk_lengths)) for ew in edge_walks]
        return {
            'semantic_path': torch.stack(semantic_path, dim=0).unsqueeze(0).numpy(),
            'anonymous_node_path': torch.stack(anonymous_node_path, dim=0).unsqueeze(0).numpy(),
            'anonymous_edge_path': torch.stack(anonymous_edge_path, dim=0).unsqueeze(0).numpy(),
            'label': self.Y[start_node].unsqueeze(0).numpy()
        }
