
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv

# 1) Choose dataset: 'Cora' or 'Pubmed'
DATASET_NAME = 'Cora'  # change to 'Pubmed' to switch

# 2) Global hyperparameter dict
HYPERPARAMS = {
    'Cora': {
        'lr': 0.01,
        'weight_decay': 5e-4,
        'hidden_channels': 64,
        'num_layers': 64,
        'dropout': 0.6,
    },
    'Pubmed': {
        'lr': 0.01,
        'weight_decay': 5e-4,
        'hidden_channels': 256,
        'num_layers': 16,
        'dropout': 0.5,
    },
}

# Load dataset
dataset = Planetoid(
    root=f'data/Planetoid/{DATASET_NAME}',
    name=DATASET_NAME,
    transform=T.NormalizeFeatures(),
)
data = dataset[0]
deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float).view(-1, 1)
data.x = deg

# GCNII model
class GCNII(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha=0.1, theta=0.5):
        super().__init__()
        self.input_lin = Linear(in_channels, hidden_channels)
        self.output_lin = Linear(hidden_channels, out_channels)
        self.convs = torch.nn.ModuleList([
            GCN2Conv(hidden_channels, alpha=alpha, theta=theta, layer=i, shared_weights=True)
            for i in range(1, num_layers + 1)
        ])
        self.dropout = torch.nn.Dropout(cfg['dropout'])
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x_0 = x = self.input_lin(x)
        for conv in self.convs:
            x = self.dropout(x)
            x = self.relu(conv(x, x_0, edge_index))
        x = self.dropout(x)
        return self.output_lin(x)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Initialize model & optimizer using the chosen hyperparameters
cfg = HYPERPARAMS[DATASET_NAME]

model = GCNII(
    in_channels=1,
    hidden_channels=cfg['hidden_channels'],
    out_channels=dataset.num_classes,
    num_layers=cfg['num_layers'],
    alpha=0.1,
    theta=0.5
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg['lr'],
    weight_decay=cfg['weight_decay'],
)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs  # [train_acc, val_acc, test_acc]

# Training loop with early stopping
best_val_acc = 0
best_test_acc = 0
patience = 100
patience_counter = 0

for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, test_acc = test()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

print(f"\nBest Val Acc: {best_val_acc:.4f}, Test Acc at Best Val: {best_test_acc:.4f}")
