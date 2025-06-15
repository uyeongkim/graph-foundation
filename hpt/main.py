import argparse
import os
import random
import tqdm
import copy
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import time
from datasets import Dataset, load_from_disk

from hypertraph_pattern.data import HypergraphPatternDataset
from model import HypergraphPatternMachine

# import warnings
# warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


parser = argparse.ArgumentParser()
parser.add_argument(
        "-t",
        "--target",
        default='cora',
        action="store",
        type=str
    )
parser.add_argument(
        "-k",
        "--num_patterns",
        default=16,
        action="store",
        type=int
    )
parser.add_argument(
        "-wn",
        "--weight_anonymous_node",
        default=1,
        action="store",
        type=float
    )
parser.add_argument(
        "-we",
        "--weight_anonymous_edge",
        default=1,
        action="store",
        type=float
    )
parser.add_argument(
        "-d",
        "--embed_dim",
        default=512,
        action="store",
        type=int
    )
parser.add_argument(
        "-ffd",
        "--feed_forward_dim",
        default=2048,
        action="store",
        type=int
    )
parser.add_argument(
        "-nh",
        "--num_heads",
        default=8,
        action="store",
        type=int
    )
parser.add_argument(
        "-bsz",
        "--batch_size",
        default=128,
        action="store",
        type=int
    )
parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-3,
        action="store",
        type=float
    )
parser.add_argument(
        "-wd",
        "--weight_decay",
        default=1e-2,
        action="store",
        type=float
    )
parser.add_argument(
        "-npl",
        "--num_pattern_layers",
        default=1,
        action="store",
        type=int
    )
parser.add_argument(
        "-ntl",
        "--num_task_layers",
        default=1,
        action="store",
        type=int
    )
parser.add_argument(
        "-dr",
        "--dropout",
        default=0.1,
        action="store",
        type=float
    )
parser.add_argument(
        "-wu",
        "--warmup",
        default=100,
        action="store",
        type=int
    )
parser.add_argument(
        "-ep",
        "--max_epoch",
        default=300,
        action="store",
        type=int
    )
parser.add_argument(
        "-es",
        "--early_stop",
        action="store_true"
    )

args = parser.parse_args()
target = args.target

# Fix random seed for reproducing.
set_seed(42) 

# Read dataset.
with open(f'dataset/{target}/data.pt', "rb") as f :
    data = pickle.load(f)
with open(f'dataset/{target}/split.pt', "rb") as f :
    splits = pickle.load(f)
    
H, X, Y= data.edge_index, data.x, data.y

# For randomwalk find node2edge, edge2node.
num_nodes = X.shape[0]
num_edges = len(torch.unique(H[1]))

node2edge = [[] for _ in range(num_nodes)]
edge2node = [[] for _ in range(num_edges)]

nodelist, edgelist = H.numpy().tolist()
edge2idx = dict()
for n, e in zip(nodelist, edgelist):
    if e not in edge2idx:
        edge2idx[e] = len(edge2idx)
    node2edge[n].append(edge2idx[e])
    edge2node[edge2idx[e]].append(n)
    
edge2str = []
for e in edge2node:
    edge2str.append(','.join(map(str, sorted(e))))
    
# k = number of patterns per node
k = args.num_patterns

# walk_lengths = lengths of random walks
walk_lengths = [2, 4, 6, 8]

# w = relative weight for anonymous walks
wn = args.weight_anonymous_node
we = args.weight_anonymous_edge

# Hyperparameters for training.
num_classes = torch.max(Y).item() + 1
batch_size = args.batch_size
label_smoothing = 0.05
lr = args.learning_rate
weight_decay = args.weight_decay
warmup_step = args.warmup
max_epoch = args.max_epoch
device = "cuda:0"

print(f"Data : {target}")
print(f"Number of nodes : {num_nodes}")
print(f"Number of hyperedges : {num_edges}")
results = []
for i in range(10):
    print(f"Split : {i}")
    cur_split = splits[i]
    train_nodes, valid_nodes, test_nodes = cur_split['train'].numpy().tolist(), cur_split['valid'].numpy().tolist(), cur_split['test'].numpy().tolist()

    # Generates random walks for each node.
    if os.path.isdir(f"temp/{args.target}/train_{i}_{k}_{walk_lengths}"):
        print("Load Patterns For Training...")
        train_data = load_from_disk(f"temp/{args.target}/train_{i}_{k}_{walk_lengths}")
    else:
        print("Generate Patterns For Training...")
        train_data = HypergraphPatternDataset(train_nodes, X, Y, node2edge, edge2node, k, walk_lengths)
        train_data = Dataset.from_list([train_data[x] for x in range(len(train_data))])
        train_data.save_to_disk(f"temp/{args.target}/train_{i}_{k}_{walk_lengths}")
        
    train_data.set_format(type="torch", columns=["semantic_path", "anonymous_node_path", "anonymous_edge_path", "label"])
        
    if os.path.isdir(f"temp/{args.target}/valid_{i}_{k}_{walk_lengths}"):
        print("Load Patterns For Validation...")
        valid_data = load_from_disk(f"temp/{args.target}/valid_{i}_{k}_{walk_lengths}")
    else:
        print("Generate Patterns For Training...")
        valid_data = HypergraphPatternDataset(valid_nodes, X, Y, node2edge, edge2node, k, walk_lengths)
        valid_data = Dataset.from_list([valid_data[x] for x in range(len(valid_data))])
        valid_data.save_to_disk(f"temp/{args.target}/valid_{i}_{k}_{walk_lengths}")
        
    valid_data.set_format(type="torch", columns=["semantic_path", "anonymous_node_path", "anonymous_edge_path", "label"])
        
    if os.path.isdir(f"temp/{args.target}/test_{i}_{k}_{walk_lengths}"):
        print("Load Patterns For Testing...")
        test_data = load_from_disk(f"temp/{args.target}/test_{i}_{k}_{walk_lengths}")
    else:
        print("Generate Patterns For Testing...")
        test_data = HypergraphPatternDataset(test_nodes, X, Y, node2edge, edge2node, k, walk_lengths)
        test_data = Dataset.from_list([test_data[x] for x in range(len(test_data))])
        test_data.save_to_disk(f"temp/{args.target}/test_{i}_{k}_{walk_lengths}")
        
    test_data.set_format(type="torch", columns=["semantic_path", "anonymous_node_path", "anonymous_edge_path", "label"])
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    train_list = []
    valid_list = []
    test_list = []
    
    for batch in train_dataloader:
        semantic = batch['semantic_path'].squeeze(1).to(device)
        anonymous_node = batch['anonymous_node_path'].squeeze(1).to(device)
        anonymous_edge = batch['anonymous_edge_path'].squeeze(1).to(device)
        labels = batch['label'].squeeze(1).to(device)
        train_list.append([semantic, anonymous_node, anonymous_edge, labels])
        
    for batch in valid_dataloader:
        semantic = batch['semantic_path'].squeeze(1).to(device)
        anonymous_node = batch['anonymous_node_path'].squeeze(1).to(device)
        anonymous_edge = batch['anonymous_edge_path'].squeeze(1).to(device)
        labels = batch['label'].squeeze(1).to(device)
        valid_list.append([semantic, anonymous_node, anonymous_edge, labels])
    
    for batch in test_dataloader:
        semantic = batch['semantic_path'].squeeze(1).to(device)
        anonymous_node = batch['anonymous_node_path'].squeeze(1).to(device)
        anonymous_edge = batch['anonymous_edge_path'].squeeze(1).to(device)
        labels = batch['label'].squeeze(1).to(device)
        test_list.append([semantic, anonymous_node, anonymous_edge, labels])
        
    model = HypergraphPatternMachine(args.embed_dim, walk_lengths,
                X.shape[1], args.feed_forward_dim, args.num_heads, args.num_pattern_layers,
                max(walk_lengths), args.num_pattern_layers, 
                args.feed_forward_dim, args.num_heads, args.num_task_layers, 
                args.dropout, wn, we, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = max_epoch)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")
    
    # Training. validation, test
    best_acc = 0
    best_loss = 1e9
    tol = 0
    for epoch in range(max_epoch):
        train_loss = 0
        train_iter = 0
        train_correct = 0
        train_cnt = 0
        model.train()
        for batch in train_list:
            t0 = time.time()
            semantic, anonymous_node, anonymous_edge, labels = batch
            t1 = time.time()
            output = model(semantic, anonymous_node, anonymous_edge)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            t2 = time.time()
            train_loss += loss.detach()
            predicted = torch.argmax(output, dim=1)
            train_correct += (predicted == labels).sum()
            train_cnt += output.shape[0]
            train_iter += 1
            t3 = time.time()
            # print(f"data load : {t1-t0}")
            # print(f"forward, backward : {t2-t1}")
            # print(f"after : {t3-t2}")
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            valid_loss = 0
            valid_iter = 0
            valid_correct = 0
            valid_cnt = 0
            model.eval()
            for batch in valid_list:
                semantic, anonymous_node, anonymous_edge, labels = batch
                output = model(semantic, anonymous_node, anonymous_edge)
                loss = criterion(output, labels)
                valid_loss += loss.detach()
                predicted = torch.argmax(output, dim=1)
                valid_correct += (predicted == labels).sum()
                valid_cnt += output.shape[0]
                valid_iter += 1
            
            train_acc = train_correct / train_cnt
            train_loss = train_loss / train_iter
            valid_acc = valid_correct / valid_cnt
            valid_loss = valid_loss / valid_iter
            
            print(f"at epoch {epoch + 1}, traning accuracy = {100 * train_acc:.4f}, training loss = {train_loss:.4f}, valid accuracy = {100 * valid_acc:.4f}, valid loss = {valid_loss:.4f}")
            
            if best_acc < valid_acc:
                print(f"parameter update : epoch {epoch + 1}")
                best_acc = valid_acc
                param = copy.deepcopy(model.state_dict())
                tol = 0
            elif best_acc == valid_acc:
                if best_loss > valid_loss:
                    print(f"parameter update : epoch {epoch + 1}")
                    param = copy.deepcopy(model.state_dict())
                    tol = 0
                else:
                    if epoch + 1 > warmup_step:
                        tol += 1
            else:
                if epoch + 1 > warmup_step:
                    tol += 1
            
            if best_loss > valid_loss:
                best_loss = valid_loss
            
            if tol == 10:
                if args.early_stop:
                    print("early stop!")
                    break

    model.load_state_dict(param)
    test_correct = 0
    test_cnt = 0
    model.eval()
    for batch in test_list:
        semantic, anonymous_node, anonymous_edge, labels = batch
        output = model(semantic, anonymous_node, anonymous_edge)
        predicted = torch.argmax(output, dim=1)
        test_correct += (predicted == labels).sum().item()
        test_cnt += output.shape[0]
            
    test_acc = test_correct / test_cnt

    print(f"test accuracy = {100 * test_acc:.4f}")
    results.append(100 * test_acc)
    
with open("result.txt", "a") as f:
    f.write(f"{args.target} (k={args.num_patterns}, wn={args.weight_anonymous_node}, we={args.weight_anonymous_edge}, d={args.embed_dim}, ffd={args.feed_forward_dim}, nh={args.num_heads}, bsz={args.batch_size}, lr={args.learning_rate}, npl={args.num_pattern_layers}, ntl={args.num_task_layers}, dr={args.dropout}, wu={args.warmup}, ep={args.max_epoch}) : accuracy = {sum(results) / len(results):.4f}, std = {np.std(results)}\n")