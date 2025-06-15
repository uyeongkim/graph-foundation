import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops, degree
import numpy as np

# --- Hyperparameter configurations from Table VI ---
CORA_CFG = {
    # 'LR': 1e-3,
    'LR': 0.05,
    'WEIGHT_DECAY': 1e-3,
    'DROPOUT_LOW': 0.0,
    'DROPOUT_HIGH': 0.9,
    'HOPS_K': 2,
    'SELF_LOOP': True,
    'THRESHOLD_T': 0.6
}
PUBMED_CFG = {
    # 'LR': 0.1,
    'LR': 0.05,
    'WEIGHT_DECAY': 1e-4,
    'DROPOUT_LOW': 0.4,
    'DROPOUT_HIGH': 0.3,
    'HOPS_K': 1,
    'SELF_LOOP': True,
    'THRESHOLD_T': 0.5
}

# --- Model dimension ---
HIDDEN_DIM = 512

# --- 1. Load dataset (uncomment the desired one) ---
# dataset = Planetoid(root='data/Planetoid', name='Cora',   split='public')
dataset = Planetoid(root='data/Planetoid', name='Pubmed', split='public')

# For example, to switch datasets, comment/uncomment the above lines.
# After loading, assign config:
config = CORA_CFG if dataset.name.lower() == 'cora' else PUBMED_CFG
LR = config['LR']
WEIGHT_DECAY = config['WEIGHT_DECAY']
DROPOUT_LOW = config['DROPOUT_LOW']
DROPOUT_HIGH = config['DROPOUT_HIGH']
HOPS_K = config['HOPS_K']
SELF_LOOP = config['SELF_LOOP']
THRESHOLD_T = config['THRESHOLD_T']

# --- 2. Prepare data and adjacency ---
data = dataset[0]

# For Pubmed, use node degree as feature; for Cora, use original features
deg_feat = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float).view(-1, 1)
data.x = deg_feat / deg_feat.max()

N = data.num_nodes
C = dataset.num_classes

# Build (normalized) adjacency matrix
torch.manual_seed(0)
edge_index = data.edge_index
if SELF_LOOP:
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
A = to_dense_adj(edge_index)[0]
A = A + torch.eye(N)
deg_vals = A.sum(1)
deg_inv_sqrt = deg_vals.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
A_norm = deg_inv_sqrt.view(-1,1) * A * deg_inv_sqrt.view(1,-1)

# --- 3. Build adjacency list for k-hop neighbors ---
adj_list = {i: A[i].nonzero(as_tuple=False).view(-1).tolist() for i in range(N)}

def get_khop_neighbors(node, k):
    visited = {node}
    frontier = {node}
    for _ in range(k):
        next_front = set()
        for u in frontier:
            next_front |= set(adj_list[u])
        frontier = next_front - visited
        visited |= frontier
    return visited

# --- 4. Neighborhood Confusion (NC) computation ---
def compute_nc(pseudo_labels, k=HOPS_K):
    nc = torch.zeros(N)
    logC = np.log(C)
    for i in range(N):
        hive = get_khop_neighbors(i, k)
        labels = pseudo_labels[list(hive)]
        counts = np.bincount(labels, minlength=C)
        p_max = counts.max() / len(hive)
        nc[i] = -np.log(p_max) / logC
    return nc

# --- 5. Mask helpers ---
def mask_rows(M, mask): return M * mask.view(-1,1)
def mask_cols(M, mask): return M * mask.view(1,-1)

# --- 6. NCGCN model definition ---
class NCGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes):
        super().__init__()
        self.W1_low  = nn.Linear(in_dim, hid_dim, bias=False)
        self.W1_high = nn.Linear(in_dim, hid_dim, bias=False)
        self.W2_low  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W2_high = nn.Linear(hid_dim, hid_dim, bias=False)
        self.Wx = nn.Linear(in_dim, hid_dim)
        self.alpha_low  = nn.Parameter(torch.tensor(0.5))
        self.alpha_high = nn.Parameter(torch.tensor(0.5))
        self.do_low  = nn.Dropout(DROPOUT_LOW)
        self.do_high = nn.Dropout(DROPOUT_HIGH)
        self.Wo = nn.Linear(hid_dim, num_classes)

    def forward(self, x, A_norm, M_low, M_high):
        H1_low  = F.relu(self.W1_low (mask_rows(A_norm, M_low)  @ x))
        H1_low  = self.do_low(H1_low)
        H1_high = F.relu(self.W1_high(mask_rows(A_norm, M_high) @ x))
        H1_high = self.do_high(H1_high)
        H2_low  = F.relu(self.W2_low (mask_cols(A_norm, M_low)  @ H1_low ))
        H2_low  = self.do_low(H2_low)
        H2_high = F.relu(self.W2_high(mask_cols(A_norm, M_high) @ H1_high))
        H2_high = self.do_high(H2_high)
        Hx = self.Wx(x)
        Ho_low  = self.alpha_low  * H2_low  + (1 - self.alpha_low ) * Hx
        Ho_high = self.alpha_high * H2_high + (1 - self.alpha_high) * Hx
        return self.Wo(Ho_low + Ho_high)

# --- 7. Training setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NCGCN(data.x.size(1), HIDDEN_DIM, C).to(device)
print(f"Feat dim: {data.x.size(1)}, Hidden dim: {HIDDEN_DIM}, Classes: {C}")
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

x = data.x.to(device)
A_norm = A_norm.to(device)
y = data.y.to(device)
train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

# NC init
nc = torch.zeros(N)
pseudo = None

# Early stopping
patience = 100
no_improve = 0
best_val = 0.0
best_state = None

# 8. Training loop
for epoch in range(1, 501):
    M_low  = (nc <= THRESHOLD_T).float().to(device)
    M_high = (nc >  THRESHOLD_T).float().to(device)

    model.train()
    opt.zero_grad()
    logits = model(x, A_norm, M_low, M_high)
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        preds = logits.argmax(1)
        val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
        if val_acc > best_val:
            best_val   = val_acc
            best_state = model.state_dict()
            pseudo     = preds.cpu().numpy()
            nc         = compute_nc(pseudo)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val: {best_val:.4f}")

# 9. Final evaluation
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out = model(x, A_norm, M_low, M_high)
    test_acc = (out[test_mask].argmax(1) == y[test_mask]).float().mean().item()
print(f"NCGCN Test Accuracy on {dataset.name} (optimal params): {test_acc:.4f}")
