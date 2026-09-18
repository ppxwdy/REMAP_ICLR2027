"""
ReMAP parameterization for unsupervised GNN MAP inference.

Supports three logits modes (ablation-friendly):
  - gnn_only:  z = rho_v(G)
  - free:      z = u
  - residual:  z = u + rho_v(G)   (default / main method)
"""
import os
# Required for torch.use_deterministic_algorithms(True) with CUDA >= 10.2 (CuBLAS).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from time import time
from itertools import chain

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, MLP, JumpingKnowledge

from utils import *
from lbp_explicit import *

torch.set_grad_enabled(True)
torch.use_deterministic_algorithms(True)

VALID_MODES = ("gnn_only", "free", "residual")
MASK_FILL_VALUE = -1e9


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def apply_label_mask(logits, label_mask, fill_value=MASK_FILL_VALUE):
    """Mask invalid / padded label dimensions before softmax."""
    if label_mask is None:
        return logits
    return logits.masked_fill(~label_mask, fill_value)


def get_temperature(step, base_temperature, temperature_schedule=None):
    """Optional temperature schedule; default is constant."""
    if temperature_schedule is None:
        return base_temperature
    if isinstance(temperature_schedule, (int, float)):
        return float(temperature_schedule)
    schedule_type = temperature_schedule.get("type", "constant")
    if schedule_type == "constant":
        return float(temperature_schedule.get("value", base_temperature))
    if schedule_type == "linear":
        start = float(temperature_schedule.get("start", base_temperature))
        end = float(temperature_schedule.get("end", base_temperature))
        total_steps = int(temperature_schedule.get("steps", 1))
        if total_steps <= 1:
            return end
        frac = min(1.0, step / max(total_steps - 1, 1))
        return start + (end - start) * frac
    raise ValueError(f"Unknown temperature schedule: {schedule_type}")


def _zero_init_mlp_last_linear(mlp):
    last_linear = None
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        torch.nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            torch.nn.init.zeros_(last_linear.bias)


class GG_Net(torch.nn.Module):
    """GNN backbone + optional free logits for ReMAP parameterization."""

    def __init__(
        self,
        num_features,
        num_classes,
        nerouns,
        dropout=0.0,
        num_layers=2,
        mask=None,
        net_key=0,
        mode="residual",
        num_nodes=None,
    ):
        super(GG_Net, self).__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        self.mask = mask
        self.dropout = dropout
        self.num_layers = num_layers
        self.mode = mode
        self.num_classes = num_classes

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_features = num_features
            else:
                in_features = nerouns
            self.lins.append(torch.nn.Linear(in_features, nerouns, bias=False))

            layer = torch.nn.Linear(in_features, nerouns, bias=False)
            if net_key == 1:
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            elif net_key == 2:
                torch.nn.init.xavier_uniform_(layer.weight)
            elif net_key == 0:
                torch.nn.init.uniform_(layer.weight)
            self.lins.append(layer)

            layer = SAGEConv(in_features, nerouns, normalize=False, bias=False)
            if net_key == 1:
                for conv_layer in [layer]:
                    if isinstance(conv_layer.lin_l, nn.Linear):
                        torch.nn.init.kaiming_uniform_(conv_layer.lin_l.weight)
                        torch.nn.init.kaiming_uniform_(conv_layer.lin_r.weight)
            elif net_key == 2:
                for conv_layer in [layer]:
                    if isinstance(conv_layer.lin_l, nn.Linear):
                        torch.nn.init.xavier_uniform_(conv_layer.lin_l.weight)
                        torch.nn.init.xavier_uniform_(conv_layer.lin_r.weight)
            elif net_key == 0:
                for conv_layer in [layer]:
                    if isinstance(conv_layer.lin_l, nn.Linear):
                        torch.geometric.nn.init.glorot(conv_layer.lin_l.weight)
                        torch.geometric.nn.init.glorot(conv_layer.lin_r.weight)

            if True and i == (num_layers - 1):
                if hasattr(layer, "lin_l") and isinstance(layer.lin_l, nn.Linear):
                    torch.nn.init.zeros_(layer.lin_l.weight)
                if hasattr(layer, "lin_r") and isinstance(layer.lin_r, nn.Linear):
                    torch.nn.init.zeros_(layer.lin_r.weight)
            self.convs.append(layer)

            if i != (num_layers - 1):
                self.bns.append(nn.BatchNorm1d(nerouns))

        self.linear_in = MLP([num_features, nerouns, num_features])
        self.linear = MLP([nerouns, nerouns * 2, num_classes])
        if mode in ("gnn_only", "residual"):
            _zero_init_mlp_last_linear(self.linear)
        self.JK = JumpingKnowledge(mode="max")

        if mode in ("free", "residual"):
            if num_nodes is None:
                raise ValueError("num_nodes is required for mode='free' or mode='residual'")
            self.free_logits = nn.Parameter(torch.zeros(num_nodes, num_classes))
        else:
            self.register_parameter("free_logits", None)

    def gnn_logits(self, data):
        """rho_v(G): raw GNN logits before free residual and masking."""
        x, edge_index = data.x, data.edge_index
        xs = []
        x = self.linear_in(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) + self.lins[i](x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.bns[i](x)
            xs.append(x)
        x = self.JK(xs)
        return self.linear(x)

    def combine_logits(self, gnn_logits):
        if self.mode == "gnn_only":
            return gnn_logits
        if self.mode == "free":
            return self.free_logits
        if self.mode == "residual":
            return self.free_logits + gnn_logits
        raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, data, temperature=1.0, return_logits=False):
        if self.mode == "free":
            logits = self.free_logits
        else:
            gnn_out = self.gnn_logits(data)
            logits = self.combine_logits(gnn_out)

        logits = apply_label_mask(logits, self.mask)
        temp = max(float(temperature), 1e-8)
        probs = F.softmax(logits / temp, dim=-1)
        if return_logits:
            return probs, logits
        return probs


def loss_func2(values, unary_energy, pair_energy, device):
    custom_loss = torch.tensor(0.0, requires_grad=True).to(device)
    for i in range(len(values)):
        custom_loss = custom_loss + torch.matmul(values[i], unary_energy[i])

    for e, vs in pair_energy.items():
        temp = torch.matmul(vs, values[e[0]])
        for i in range(1, len(e)):
            temp = torch.matmul(temp, values[e[i]])
        custom_loss = custom_loss + temp

    return custom_loss


def create_variable_class_mask(num_nodes, states, device):
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


def build_node_to_cliques(nodes, cliques):
    mapping = {int(i): [] for i in nodes}
    for clique in cliques:
        for node in clique:
            mapping[int(node)].append(clique)
    return mapping


def _factor_conditional_cost_np(vs, clique, node_i, p):
    """Expected factor cost as a function of x_{node_i}, other nodes using p."""
    vs = np.asarray(vs)
    k = len(clique)
    if k == 1:
        return np.asarray(vs, dtype=np.float64)
    letters = "ijklmnopqrstuvwxyz"
    if k > len(letters):
        raise ValueError(f"clique order {k} exceeds einsum label budget")
    subs = "".join(letters[:k])
    keep = None
    rhs = []
    ops = [vs]
    for d in range(k):
        var = clique[k - 1 - d]
        if var == node_i:
            if keep is not None:
                raise ValueError(f"repeated node {node_i} in clique {clique}")
            keep = letters[d]
        else:
            rhs.append(letters[d])
            vec = p[var]
            dim = vs.shape[d]
            if vec.shape[0] != dim:
                vec = vec[:dim]
            ops.append(vec)
    if keep is None:
        raise ValueError(f"node {node_i} not in clique {clique}")
    if not rhs:
        return np.asarray(vs, dtype=np.float64)
    return np.einsum(f"{subs},{','.join(rhs)}->{keep}", *ops)


def round_argmax(probs):
    return torch.argmax(probs, dim=1).detach().cpu().numpy()


def round_conditional_expectation(
    probs, unary_energy, pairwise_energy, node_to_cliques, n_states, nodes
):
    """Sequential CE rounding: E(x_CE) <= F(p) for any variable order."""
    p = probs.detach().cpu().numpy().astype(np.float64, copy=True)
    n = p.shape[0]
    assignment = np.zeros(n, dtype=np.int64)
    for i in nodes:
        i = int(i)
        g = np.array(unary_energy[i], dtype=np.float64, copy=True)
        s_i = int(n_states[i])
        for clique in node_to_cliques[i]:
            slice_i = _factor_conditional_cost_np(pairwise_energy[clique], clique, i, p)
            g[:s_i] += np.asarray(slice_i, dtype=np.float64)[:s_i]
        if s_i < g.shape[0]:
            g[s_i:] = np.inf
        a = int(np.argmin(g))
        assignment[i] = a
        p[i] = 0.0
        p[i, a] = 1.0
    return assignment


def decode_assignment(
    probs,
    rounding,
    unary_energy,
    pairwise_energy,
    node_to_cliques,
    n_states,
    nodes,
):
    if rounding == "argmax":
        return round_argmax(probs)
    if rounding == "ce":
        return round_conditional_expectation(
            probs, unary_energy, pairwise_energy, node_to_cliques, n_states, nodes
        )
    raise ValueError(f"Unknown rounding method: {rounding!r}")


def create_net(
    nodes_num,
    gnn_hypers,
    opt_params,
    torch_device,
    torch_dtype,
    net_type=0,
    mode="residual",
):
    num_features = gnn_hypers["num_features"]
    num_classes = gnn_hypers["num_classes"]
    nerouns = gnn_hypers["nerouns"]
    dropout = gnn_hypers["dropout"]
    num_layers = gnn_hypers.get("num_layers", 2)

    net = GG_Net(
        num_features,
        num_classes,
        nerouns,
        dropout,
        num_layers=num_layers,
        net_key=net_type,
        mode=mode,
        num_nodes=nodes_num,
    )
    net = net.to(torch_device)

    embed = nn.Embedding(nodes_num, num_features)
    embed = embed.type(torch_dtype).to(torch_device)

    # Includes GNN + free_logits (when present) + embedding.
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, optimizer, embed


def train(
    nodes,
    edges,
    cliques,
    unary_energy,
    pairwise_energy,
    masks,
    net,
    num_iter,
    optimizer,
    embed,
    device,
    feature_num,
    tol=0.001,
    patience=1000,
    seed=666,
    mode="residual",
    temperature=1.0,
    temperature_schedule=None,
    rounding="ce",
):
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
    temperatures_used = []

    best_out = None
    best_loss = sys.float_info.max
    prev_loss = sys.float_info.max
    count = 0

    unary_energy_soft = {}
    pairwise_energy_soft = {}

    for k, v in unary_energy.items():
        t = torch.tensor(v, dtype=torch.float32, requires_grad=True).to(device)
        unary_energy_soft[k] = t

    for k, v in pairwise_energy.items():
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
    t4s = []
    t2s = []
    iter_count = 0

    best_rounded_energy = sys.float_info.max
    best_rounded_assignment = None
    final_relaxed_loss = None
    final_rounded_energy = None
    node_to_cliques = build_node_to_cliques(nodes, cliques)
    n_states = masks.sum(dim=1).detach().cpu().numpy().astype(int)

    for iter_idx in tqdm(range(num_iter)):
        t1 = time()
        optimizer.zero_grad()

        step_temperature = get_temperature(iter_idx, temperature, temperature_schedule)
        temperatures_used.append(step_temperature)

        out = net(data, temperature=step_temperature)

        t3 = time()
        loss = loss_func2(out, unary_energy_soft, pairwise_energy_soft, device)
        t4 = time() - t3
        losses.append(loss)

        with torch.no_grad():
            v_round = decode_assignment(
                out,
                rounding,
                unary_energy,
                pairwise_energy,
                node_to_cliques,
                n_states,
                nodes,
            )
            rounded_energy = loss_clique(v_round, nodes, cliques, unary_energy, pairwise_energy)
            if rounded_energy < best_rounded_energy:
                best_rounded_energy = rounded_energy
                best_rounded_assignment = v_round.copy()

        if (abs(loss - prev_loss) <= tol) | ((loss - prev_loss) > 0):
            count += 1
        else:
            count = 0

        loss.backward()
        optimizer.step()
        prev_loss = loss.item()
        final_relaxed_loss = prev_loss

        if iter_count == 0:
            print(
                f"Iter {iter_idx} mode={mode} rounding={rounding} T={step_temperature:.4f} "
                f"loss={loss.item():.6f} rounded={rounded_energy:.6f} "
                f"best_rounded={best_rounded_energy:.6f}"
            )
            iter_count += 1
        else:
            iter_count += 1
            if iter_count == 10:
                iter_count = 0

        x = embed(torch.tensor([i for i in range(len(nodes))]).to(device)).to(device)
        data = Data(x=x, edge_index=edge_index.contiguous())
        data = data.to(device)

        current_time = time() - start

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_out = out
            best_ans.append(best_loss)
            best_time = current_time

        if print_time:
            if t_20 and current_time > 20:
                key_point.append(best_loss)
                t_20 = False
            elif t_1200 and current_time > 1200:
                key_point.append(best_loss)
                t_1200 = False
            elif t_3600 and current_time > 3600:
                key_point.append(best_loss)
                t_3600 = False
                print_time = False

        times.append(current_time)
        if iter_idx == 99:
            if current_time > 1200 and total >= 100:
                break
            else:
                total += 100

        t2 = time() - t1
        t2s.append(t2)
        t4s.append(t4)

    loss_curve, v = None, False
    wall_clock_time = times[-1] if times else 0.0
    num_steps = len(losses)

    print(
        "Time used in loss calculation:",
        np.mean(t4s) if t4s else 0.0,
        "Per round:",
        np.mean(t2s) if t2s else 0.0,
    )

    metrics = {
        "mode": mode,
        "seed": seed,
        "temperature": temperature,
        "temperature_schedule": temperature_schedule,
        "temperature_min": min(temperatures_used) if temperatures_used else temperature,
        "temperature_max": max(temperatures_used) if temperatures_used else temperature,
        "num_steps": num_steps,
        "wall_clock_time": wall_clock_time,
        "final_relaxed_loss": final_relaxed_loss,
        "best_relaxed_loss": best_loss if best_loss < sys.float_info.max else None,
        "rounding": rounding,
        "rounded_discrete_energy": None,
        "best_rounded_energy": best_rounded_energy if best_rounded_energy < sys.float_info.max else None,
    }

    if key:
        print("No result since Nan or Inf appears!")
        return loss_curve, v, metrics

    for i in range(len(key_point)):
        print(f"Provisional Best Result at {key_ts[i]}s : {key_point[i]}")

    if best_rounded_assignment is not None:
        v = best_rounded_assignment
    elif best_out is not None:
        v = decode_assignment(
            best_out,
            rounding,
            unary_energy,
            pairwise_energy,
            node_to_cliques,
            n_states,
            nodes,
        )
    else:
        v = False

    if v is False:
        return loss_curve, v, metrics

    rounded_energy = loss_clique(v, nodes, cliques, unary_energy, pairwise_energy)
    final_rounded_energy = rounded_energy
    metrics["rounded_discrete_energy"] = rounded_energy
    metrics["best_rounded_energy"] = best_rounded_energy

    print("best relaxed loss:", best_loss)
    print("final relaxed loss:", final_relaxed_loss)
    print("rounded discrete energy:", rounded_energy)
    print("best rounded energy:", best_rounded_energy)
    print("rounding:", rounding)
    print("mode:", mode)
    print("seed:", seed)
    print("temperature:", temperature)
    print("temperature schedule:", temperature_schedule)
    print("num optimization steps:", num_steps)
    print("Time used:", wall_clock_time)
    print("Best answer get at ", best_time)

    loss_curve = [l.item() for l in losses]
    return loss_curve, v, metrics


def check_feasi(out, states):
    count = 0
    for i, o in enumerate(out):
        if o >= states[i]:
            count += 1
    print(f"{count} vars are infeasible!")
