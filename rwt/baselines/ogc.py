import argparse
import os.path as osp
import time
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import one_hot, degree

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support.*')

# Per-dataset hyperparameters\
HYPERPARAMS = {
    'cora': {
        'decline': 0.9,
        'eta_sup': 0.001,
        'eta_W': 0.5,
        'beta': 0.1,
        'max_sim_tol': 0.995,
        'max_patience': 2,
    },
    'pubmed': {
        'decline': 0.9,
        'eta_sup': 0.01, # 0.1 / 0.05 / 0.01 / 0.001
        'eta_W': 0.5,
        'beta': 0.1,
        'max_sim_tol': 0.995,
        'max_patience': 2,
    },
}

# Map lowercase keys to Planetoid names
DATASET_NAME_MAP = {
    'cora': 'Cora',
    'pubmed': 'PubMed',
}

parser = argparse.ArgumentParser()
# CLI arguments\parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    help="Dataset to use ('Cora' or 'PubMed')")
args = parser.parse_args()

dataset_key = args.dataset.lower()
if dataset_key not in HYPERPARAMS:
    raise ValueError(f"No hyperparameters configured for dataset '{args.dataset}'")
params = HYPERPARAMS[dataset_key]

# Set globals from hyperparameters
decline      = params['decline']
eta_sup      = params['eta_sup']
eta_W        = params['eta_W']
beta         = params['beta']
max_sim_tol  = params['max_sim_tol']
max_patience = params['max_patience']

# Device and paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')

# Transforms	
transform = T.Compose([
    T.NormalizeFeatures(),
    T.GCNNorm(),
])

# Load dataset
dataset = Planetoid(path, name=DATASET_NAME_MAP[dataset_key], transform=transform)
data = dataset[0].to(device)

# Feature: node degree
deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float).view(-1, 1)
data.x = deg / deg.max()

# Convert to sparse tensor
data = T.ToSparseTensor(layout=torch.sparse_csr)(data).to(device)

# Prepare labels and masks
y_one_hot = one_hot(data.y, dataset.num_classes)
data.trainval_mask = data.train_mask | data.val_mask
S = torch.diag(data.train_mask).float().to_sparse()
I_N = torch.eye(data.num_nodes).to_sparse(layout=torch.sparse_csr).to(device)

# Lazy random walk adjacency
lazy_adj = beta * data.adj_t + (1 - beta) * I_N

class LinearNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(num_features, num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.W(x)

    @torch.no_grad()
    def test(self, U: Tensor, y_one_hot: Tensor, data: Data):
        self.eval()
        out = self(U)

        loss = F.mse_loss(
            out[data.trainval_mask],
            y_one_hot[data.trainval_mask],
        )

        accs = []
        pred = out.argmax(dim=-1)
        for _, mask in data('trainval_mask', 'test_mask'):
            accs.append(float((pred[mask] == data.y[mask]).sum() / mask.sum()))

        return float(loss), accs[0], accs[1], pred

    def update_W(self, U: Tensor, y_one_hot: Tensor, data: Data):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta_W)
        self.train()
        optimizer.zero_grad()
        pred = self(U)
        loss = F.mse_loss(pred[data.trainval_mask], y_one_hot[
            data.trainval_mask,
        ], reduction='sum')
        loss.backward()
        optimizer.step()
        return self(U).data, self.W.weight.data

# Instantiate model
model = LinearNeuralNetwork(
    num_features=1,
    num_classes=dataset.num_classes,
    bias=False,
).to(device)

def update_U(U: Tensor, y_one_hot: Tensor, pred: Tensor, W: Tensor):
    global eta_sup

    # Smoothness update
    U = lazy_adj @ U

    # Supervised update
    dU_sup = 2 * (S @ (-y_one_hot + pred)) @ W
    U = U - eta_sup * dU_sup

    # Decay learning rate
    eta_sup = eta_sup * decline
    return U

def ogc() -> float:
    U = data.x
    _, _, last_acc, last_pred = model.test(U, y_one_hot, data)

    patience = 0
    for i in range(1, 65):
        # Update W
        pred, W = model.update_W(U, y_one_hot, data)

        # Update U
        U = update_U(U, y_one_hot, pred, W)

        # Evaluate
        loss, trainval_acc, test_acc, pred = model.test(U, y_one_hot, data)
        print(f'Epoch: {i:02d}, Loss: {loss:.4f}, '
              f'Train+Val Acc: {trainval_acc:.4f} Test Acc {test_acc:.4f}')

        # Early stopping based on prediction similarity
        sim_rate = float((pred == last_pred).sum()) / pred.size(0)
        if sim_rate > max_sim_tol:
            patience += 1
            if patience > max_patience:
                break

        last_acc, last_pred = test_acc, pred

    return last_acc

if __name__ == '__main__':
    start_time = time.time()
    test_acc = ogc()
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Total Time: {time.time() - start_time:.4f}s')
