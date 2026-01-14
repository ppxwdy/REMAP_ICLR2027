import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP, GraphConv, JumpingKnowledge, NNConv
from torch_geometric.utils import add_random_edge, dropout_edge
import networkx as nx
import numpy as np
from time import time
import random
from itertools import chain, islice
from utils import *

import sys


import matplotlib.pyplot as plt



torch.set_grad_enabled(True)
torch.use_deterministic_algorithms(True) 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



class GG_Net(torch.nn.Module):
    
    def __init__(self, num_features, num_classes, nerouns, dropout=0.0, num_layers=5, mask=None):
        super(GG_Net, self).__init__()
        self.mask = mask
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_features = num_features
            else:
                in_features = nerouns

            layer = torch.nn.Linear(in_features, nerouns, bias=False)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.lins.append(layer)
            layer = SAGEConv(in_features, nerouns, normalize=False, bias=False)
            for layer in [layer]:
                if isinstance(layer.lin_l, nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.lin_l.weight)
                    torch.nn.init.kaiming_uniform_(layer.lin_r.weight)
            self.convs.append(layer)
            
            if i != (num_layers - 1):
                self.bns.append(nn.BatchNorm1d(nerouns))
      
        self.linear = MLP([nerouns, nerouns*2, num_classes])
        # self.softmax = torch.nn.Softmax()
        self.JK = JumpingKnowledge(mode='max')
        
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        xs = []
        x0 = data.x 
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) + self.lins[i](x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
                x = self.bns[i](x)
            xs.append(x)
            
        x = self.JK(xs)
        x = self.linear(x)
        
        if self.mask is not None:

            x[~self.mask] = float('-inf')
            
        return F.softmax(x, dim=-1)
    


def loss_func2(values, nodes, cliques, unary_energy, pair_energy, device):
    custom_loss = torch.tensor(0.0, requires_grad=True).to(device)
    for i in range(len(values)):

        custom_loss = custom_loss + torch.matmul(values[i], unary_energy[i])

    key = True
    temp_sum = 0
    for e, vs in pair_energy.items():

        temp = torch.matmul(vs, values[e[0]])

        for i in range(1, len(e)):
            temp = torch.matmul(temp, values[e[i]])
                # count2 += 1
        custom_loss = custom_loss + temp

 
    return custom_loss
    
def create_variable_class_mask(num_nodes, states, device):
    """
    Creates a mask for nodes with different numbers of classes
    
    Args:
        num_nodes: Total number of nodes
        class_counts: List of number of classes for each node
        max_classes: Maximum number of classes (padding dimension)
    
    Returns:
        torch.Tensor: Boolean mask of shape [num_nodes, max_classes]
    """
    max_classes = max(states)
    
    mask = torch.zeros(num_nodes, max_classes, dtype=torch.bool).to(device)
    
    for node_idx, num_classes in enumerate(states):
        mask[node_idx, :num_classes] = True
    
    return mask


   
def loss_clique(v, nodes, cliques, unary_energy, clique_energy):
    loss = 0
    for i in nodes:

        loss += unary_energy[i][v[i]]
    for clique in cliques:
        idx = []
        for n in clique:
            idx.append(v[n])
        loss += clique_energy[clique][tuple(idx[::-1])]

    return loss   


def create_net(nodes_num, gnn_hypers, opt_params, torch_device, torch_dtype):
    
    num_features = gnn_hypers["num_features"]
    num_classes = gnn_hypers["num_classes"]
    nerouns = gnn_hypers["nerouns"]
    dropout = gnn_hypers["dropout"]
    
    net = GG_Net(num_features, num_classes, nerouns, dropout)
    net = net.to(torch_device)
    
    embed = nn.Embedding(nodes_num, num_features)
    embed = embed.type(torch_dtype).to(torch_device)
    
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)

    return net, optimizer, embed



def train(nodes, edges, cliques, unary_energy, pairwise_energy, masks, net, num_iter, 
        optimizer, embed, device, feature_num, tol=0.1, patience=1000):
        

    x = embed(torch.tensor([i for i in range(len(nodes))]).to(device)).to(device)

    edge_index = [[], []]
    for edge in edges:
        edge_index[0].append(edge[0])
        edge_index[0].append(edge[1])
        edge_index[1].append(edge[1])
        edge_index[1].append(edge[0])
        
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.contiguous())
    data = data.to(device)
    
    losses = []
    
    best_out = None
    best_loss = sys.float_info.max
    prev_loss = sys.float_info.max
    count = 0
    
    unary_energy_soft = {}
    pairwise_energy_soft = {}
    
    
    for k,v in unary_energy.items():
        t = torch.tensor(v, dtype=torch.float32, requires_grad=True).to(device)

        unary_energy_soft[k] = t
    
    for k,v in pairwise_energy.items():
        t = torch.tensor(v, dtype=torch.float32, requires_grad=True).to(device)

        pairwise_energy_soft[k] = t

    key = False
    start = time()
    times = []
    best_ans = []
    
    print_time = True
    t_20 = True
    t_1200 = True 
    t_3600 = True
    key_point = []
    key_ts = [20, 1200, 3600]

    total = 0
    
    change_lr_count = 0
    iter_count = 0
    t4s = []
    t2s = []
    iter_count = 0
    for iter in tqdm(range(num_iter)):
        t1 = time()
        optimizer.zero_grad()

        out = net(data)
        t3 = time()
        loss = loss_func2(out, nodes, cliques, unary_energy_soft, pairwise_energy_soft, device)
        t4 = time()-t3
        losses.append(loss)

        
        if (abs(loss - prev_loss) <= tol) | ((loss - prev_loss) > 0):
            count += 1
        else:
            count = 0
        # print(loss.item())
        if count >= patience or torch.isnan(loss):
            print(f'Stopping early on iteration {iter} (patience: {patience})')
            key = True
            break
            
        loss.backward()
        optimizer.step()

        x = embed(torch.tensor([i for i in range(len(nodes))]).to(device)).to(device)
        data = Data(x=x,  edge_index=edge_index.contiguous())
        data = data.to(device)
        
        current_time = time() - start
        if iter_count == 0:
            print("Iter", iter, " loss:", loss)
            iter_count += 1
        else:
            iter_count += 1
            if iter_count == 10:
                iter_count = 0
                
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_out = out
            best_ans.append(best_loss)
            best_time = current_time
        
        if print_time:
            if t_20 and current_time > 20:
                # print("sol at 20s:", best_loss)
                key_point.append(best_loss)
                t_20 = False
            elif t_1200 and current_time > 1200:
                # print("sol at 1200s:", best_loss)
                key_point.append(best_loss)
                t_1200 = False
            elif t_3600 and current_time > 3600:
                # print("sol at 3600s:", best_loss)
                key_point.append(best_loss)
                t_3600 = False
                print_time = False
                
        times.append(current_time)
        if iter == 99:
            if current_time > 1200 and total >= 100:
                break
            else:
                total += 100
                iter == 0
        t2 = time()-t1

        t2s.append(t2)
        t4s.append(t4)
    
    print("Time used in loss calculation:", np.mean(t4s), "Per round:", np.mean(t2s), " About ", np.mean(t4s)/np.mean(t2s), " %")
    loss_curve, v = None, False

    
    if key:
        print("No result since Nan or Inf appears!")
        return loss_curve, v
    else:
        for i in range(len(key_point)):
            print(f"Provisional Best Result at {key_ts[i]}s : {key_point[i]}")
            
        v = torch.argmax(best_out, 1).detach().cpu().numpy()
        # print(v)
        print("best loss:", best_loss)
        print("actual loss:", loss_clique(v, nodes, cliques, unary_energy, pairwise_energy).item())

        loss_curve = []
        for l in losses:
            loss_curve.append(l.item())

        print("Time used:", times[-1])
        print("Best answer get at ", best_time)
        return loss_curve, v
    

    
    

def check_feasi(out, states):
    count = 0
    for i, o in enumerate(out):
        if o >= states[i]:
            count += 1
    print(f"{count} vars are infeasible!")
