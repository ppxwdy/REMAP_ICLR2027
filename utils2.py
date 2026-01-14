import networkx as nx
import numpy as np
import os
from tqdm import tqdm

def er_graph_obs(n, p=0.6, seed=6):
    np.random.seed(seed)
    g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    
    # random state_nums
    states = [i for i in range(10, 16)]
    state_nums = {gnodes[i]: np.random.choice(states) for i in range(n)}

    # value for observe nodes    
    observations = {}
    
    for node in gnodes:
 
        states = [i for i in range(1, state_nums[node]+1)]
        observations[node] = np.random.choice(states)

    S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    if len(S) > 1:
        nodee = []
        for i in range(len(S)):
            nodee.append(list(S[i].nodes)[0])
            
        for j in range(len(nodee)-1):
            n1, n2 = min(nodee[j], nodee[j+1]), max(nodee[j], nodee[j+1])
            g.add_edge(n1, n2)
    # random_spanning_tree = nx.random_spanning_tree(g, seed=seed)
    # g = random_spanning_tree
    return state_nums, observations, g


def grid_obs(n,m, min_state_num=1, max_state_num=10, seed=6, p=1):
    g = nx.Graph()
    nodes = [i for i in range(n*m)]
    g.add_nodes_from(nodes)

    np.random.seed(seed)
    # random state_nums
    # states = [i for i in range(10, 16)]
    states = [min_state_num]
    state_nums = {nodes[i]: np.random.choice(states) for i in range(n*m)}

    observations = {}
    for row in range(n-1):
        for col in range(m-1):
            node = row*m + col
            right = node + 1
            below = node + m
            if np.random.rand() < p:
                g.add_edge(node, right)
                g.add_edge(node, below)
            # observation
            states = [i for i in range(1, state_nums[node]+1)]
            observations[node] = np.random.choice(states)

   
    for row in range(n-1):
        node = (row + 1) * m - 1
        below = (node + m)
        if np.random.rand() < p:
            g.add_edge(node, below)
        # observation
        states = range(1, state_nums[node]+1)
        observations[node] = np.random.choice(states)
    
    for col in range(m-1):
        node = (n-1)*m + col
        right = node + 1
        if np.random.rand() < p:
            g.add_edge(node, right)
        # observation
        states = range(1, state_nums[node]+1)
        observations[node] = np.random.choice(states)

    node = n*m - 1 
    states = range(1, state_nums[node]+1)
    observations[node] = np.random.choice(states)
    
    # print(observations)
    return state_nums, observations, g



# node degree = d, then the input p = d/(|V| - 1)
def er_graph(n, p=0.6, min_state=10, max_state=11, seed=6):
    np.random.seed(seed)
    g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    
    # random state_nums
    states = [state for state in range(min_state, max_state)]
    state_nums = {gnodes[i]: np.random.choice(states) for i in range(n)}

    # value for observe nodes    

    for node in gnodes:
        states = [i for i in range(1, state_nums[node]+1)]

    S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    if len(S) > 1:
        nodee = []
        for i in range(len(S)):
            nodee.append(list(S[i].nodes)[0])
            
        for j in range(len(nodee)-1):
            n1, n2 = min(nodee[j], nodee[j+1]), max(nodee[j], nodee[j+1])
            g.add_edge(n1, n2)

    return state_nums,  g



def grid(n,m, min_state=10, max_state=11, seed=6):
    g = nx.Graph()
    nodes = [i for i in range(n*m)]
    g.add_nodes_from(nodes)

    np.random.seed(seed)

    states = [state for state in range(min_state, max_state)]
    state_nums = {nodes[i]: np.random.choice(states) for i in range(n*m)}

    for row in range(n-1):
        for col in range(m-1):
            node = row*m + col
            right = node + 1
            below = node + m
            g.add_edge(node, right)
            g.add_edge(node, below)
    
    for row in range(n-1):
        node = (row + 1) * m - 1
        below = (node + m)
        g.add_edge(node, below)
    
    for col in range(m-1):
        node = (n-1)*m + col
        right = node + 1
        g.add_edge(node, right)

    return state_nums,  g



def energy_generator(g, state_nums, seed=6, padding=False):
    np.random.seed(seed)
    unary_energy = {node:np.random.rand(state_nums[node]) for node in list(g.nodes)}
    pair_energy = {(e1, e2):np.random.rand(state_nums[e1], state_nums[e2]) for (e1, e2) in list(g.edges)}
    unary_energy_p, pair_energy_p = {}, {}
    masks ={}
    max_state = 0
    for k,v in state_nums.items():
        max_state = max(max_state, v)
    if padding:
        
        for node in list(g.nodes):
            ue = np.random.rand(max_state)
            ue[:state_nums[node]] = unary_energy[node][:]
            v = np.max(unary_energy[node][:])
            for i in range(max_state):
                if i >= state_nums[node]:
                    ue[i] = v#np.sum(ue)*10
            unary_energy_p[node] = ue
            out = np.ones(max_state)
            # masks[node] = np.ones_like(out, dtype=bool)
            # masks[node][state_nums[node]-max_state:] = False
            
        for e1, e2 in list(g.edges):
            pe = np.random.rand(max_state, max_state)
            # pe[:state_nums[e1]][:state_nums[e2]] = pair_energy[(e1, e2)][:, :]
            v = np.max(pair_energy[(e1, e2)])
            for i in range(max_state):
                for j in range(max_state):
                    if i >= state_nums[e1] or j >= state_nums[e2]:
                        pe[i][j] = v#np.sum(pair_energy[(e1, e2)]) * 10
                    else:
                        pe[i][j] = pair_energy[(e1, e2)][i][j]
            pair_energy_p[(e1, e2)] = pe
    else:
        unary_energy_p = unary_energy
        pair_energy_p = pair_energy
    return unary_energy, pair_energy, unary_energy_p, pair_energy_p, max_state, masks
    

def pad_uai(nodes, edges, states, unary_energy, pair_energy):
    unary_energy_p, pair_energy_p = {}, {}
    masks ={}
    max_state = 0
    state_nums = {}
    for k,v in enumerate(states):
        max_state = max(max_state, v)
        state_nums[nodes[k]] = v
        
    for node in list(nodes):
            ue = np.random.rand(max_state)
            ue[:state_nums[node]] = unary_energy[node][:]
            v = np.max(unary_energy[node][:])
            v2 = np.min(unary_energy[node][:])
            for i in range(max_state):
                if i >= state_nums[node]:
                    ue[i] = v#np.sum(ue)*10
            unary_energy_p[node] = ue
            out = np.ones(max_state)
            # masks[node] = np.ones_like(out, dtype=bool)
            # masks[node][state_nums[node]-max_state:] = False
            
    for e1, e2 in list(edges):
        pe = np.random.rand(max_state, max_state)
        # pe[:state_nums[e1]][:state_nums[e2]] = pair_energy[(e1, e2)][:, :]
        v = np.max(pair_energy[(e1, e2)])
        v2 = np.min(pair_energy[(e1, e2)])

        for i in range(max_state):
            for j in range(max_state):
                if i >= state_nums[e1] or j >= state_nums[e2]:
                    pe[i][j] = v#np.sum(pair_energy[(e1, e2)]) * 10
                else:

                    pe[i][j] = pair_energy[(e1, e2)][i][j]
        pair_energy_p[(e1, e2)] = pe

    return unary_energy_p, pair_energy_p, max_state, masks
    


def save_model(model, path, file):
    """_summary_

    Args:
        model (_type_): (nodes, edges, states, unary_energy, pair_energy)
        path (_type_): _description_
        file (_type_): _description_
    """
    f = open(path + file + ".txt",'w')
    nodes, edges, states, unary_energy, pair_energy = model
    row = 0
    idx = 0
    idx2 = 0

    while 1:
        
        if row == 0:
            f.write("MARKOV\n")
        elif row == 1:
            f.write(str(len(nodes)))
            f.write("\n")
        elif row == 2:
            for i in range(len(nodes)):
                f.write(f"{states[i]} ")
            f.write("\n")
        elif row == 3:
            f.write(f"{len(edges)+len(nodes)}\n")
        elif row > 3 and row < 4+len(nodes):
            f.write(f"1 {nodes[row-4]}\n")
        elif row > 3 and row < 4+len(nodes) + len(edges):
            f.write(f"2 {edges[row-4-len(nodes)][0]} {edges[row-4-len(nodes)][1]}\n")
        elif idx < len(nodes):
            f.write("\n")
            f.write(f"{states[idx]}\n")
            for i in range(len(unary_energy[idx])):
                f.write(f"{unary_energy[idx][i]} ")
            f.write("\n")
            idx += 1
        elif idx2 < len(edges):
            f.write("\n")
            f.write(f"{states[edges[idx2][0]]*states[edges[idx2][1]]}\n")
            for i in range(len(pair_energy[edges[idx2]])):
                for j in range(len(pair_energy[edges[idx2]][0])):
                    f.write(f"{pair_energy[edges[idx2]][i][j]} ")
            f.write("\n")
            idx2 += 1
        else:
            break
        row += 1



def dfs(clique, node_idx, states, table, idx_set, max_state, max_v, exceed=False):
    temp = []
    if node_idx == 0:
        for i in range(max_state):
            if i < states[clique[node_idx]] and exceed == False:
                idx_set[node_idx] = i
                temp.append(table[tuple(idx_set)])
            else:
                idx_set[node_idx] = i
                temp.append(max_v)
        idx_set[node_idx] = 0
    else:
        for i in range(max_state):

            if i < states[clique[node_idx]] and exceed == False:
                
                idx_set[node_idx] = i
                temp.append(dfs(clique, node_idx-1, states, table, idx_set, max_state, max_v, exceed))
            else:
                idx_set[node_idx] = i
                temp.append(dfs(clique, node_idx-1, states, table, idx_set, max_state, max_v, True))
        idx_set[node_idx] = 0
    return temp


def get_mat(clique, states, data, max_state):
    ll = len(clique)
    
    idx_set = [0 for i in range(ll)]
    array = {}

    for d in data:
        array[tuple(idx_set)] = d
        for i in range(ll-1, -1, -1):
            
            if idx_set[i] < states[clique[i]] - 1:
                idx_set[i] += 1
                break
            elif idx_set[i] == states[clique[i]] - 1:
                idx_set[i] = 0
    # print(array)
    max_v = np.max(data)
    # print(array)
    idx_set = [0 for i in range(ll)]
    node_idx = len(clique) - 1
    rtn = dfs(clique, node_idx, states, array, idx_set, max_state, max_v)
    
    return np.array(rtn)    
        
# print(get_mat([0, 1, 2, 3], {0:2, 1:2, 2:2, 3:2}, [1, 9, 5, 13, 3, 11, 7, 15, 2, 10, 6, 14, 4, 12, 8, 16]))        






def read_clique_model2(filenamae):
    f = open(filenamae, 'r')
    lines = f.readlines()
    f.close()
    
    row = 0

    nodes = []
    cliques = []
    edges = set()
    states = []
    
    unary_energy = {}
    pair_energy = {}
    
    count = 0
    idx = 0
    target_count = 0
    current_count = 0
    temp_energy_lis = []
    
    max_state = 0
    
    order = {}
    o_i = 0
    p_o_u = {}
    multi = False
    m = 0
    repeat = 0
    zero_e = False

    for line in lines:
        # print(line=='\n')
        if line == '\n':
            row += 1
            # print("in")
            continue
        if row == 0:
            pass
        elif row == 1:
            n = int(line)
        elif row == 2:
            states = [int(i) for i in line.split()]
            for node in range(len(states)):
                nodes.append(node)
        
            max_state = max(states)
        elif row == 3:
            m = int(line)
        elif row > 3 and row < 4+m:
            line_s = line.split()
            temp = []
            if len(line_s) == 1:
                m -= 1
                row += 1
                continue
            if len(line_s) == 2:
                order[o_i] = int(line_s[1])
                p_o_u[o_i] = 0 
                o_i += 1
            else:
                for i in range(1, len(line_s)):
                    temp.append(int(line_s[i]))
                    for j in range(i+1, len(line_s)):
                        edges.add((min(int(line_s[i]), int(line_s[j])), max(int(line_s[i]), int(line_s[j]))))

                cliques.append(tuple(temp))
                order[o_i] = tuple(temp)
                p_o_u[o_i] = 1
                o_i += 1
        elif idx < o_i:

            if count == 1:
                
                temp_lllis = [float(i) for i in lines[row].split()]
                
                ll = len(temp_lllis)
                current_count += ll
    
                    
                # for i in range(len(temp_lllis)):
                #     temp_lllis[i] = -np.log(0.01) if temp_lllis[i] == 0 else -np.log(float(temp_lllis[i]))
                for i in range(len(temp_lllis)):
                    if temp_lllis[i] == 0:
                        temp_lllis[i] = -np.log(0.001)
                        # print("here", row,  -np.log(1924402196383962065010688.000000 ), order[idx])
                        zero_e = True
                    else:
                        temp_lllis[i] = -np.log(float(temp_lllis[i]))
                temp_energy_lis += temp_lllis

                
                
                # print(row, idx, current_count, target_count, temp_lllis)
                if current_count == target_count:
                    if p_o_u[idx] == 0:

                        if order[idx] not in unary_energy:
                            unary_energy[order[idx]] = [temp_energy_lis]
                        else:
                            unary_energy[order[idx]].append(temp_energy_lis)               
                    else:
                        if order[idx] not in pair_energy:
                            pair_energy[order[idx]] = [get_mat(order[idx], states, temp_energy_lis, max_state)]
                        else:
                            pair_energy[order[idx]].append(get_mat(order[idx], states, temp_energy_lis, max_state))
                    temp_energy_lis = []
                    idx += 1
                    count = 0
                    current_count = 0
            else:    

                count += 1
                target_count = int(line)

        row += 1

    
    for k, v_lis in pair_energy.items():
        a = np.copy(v_lis[0])
        if len(v_lis) > 1:
            for v in v_lis:
                a = np.maximum(a, v)
        pair_energy[k] = a
    
    for k, v_lis in unary_energy.items():
        a = np.copy(v_lis[0])
        if len(v_lis) > 1:
            for v in v_lis:
                a = np.maximum(a, v)
        unary_energy[k] = np.append(a, [np.max(a) for i in range(max_state - len(a))])
                
    for n in nodes:
        if n not in unary_energy:
            unary_energy[n] = np.zeros(max_state)
            
    # for k,v_lis in unary_energy.items():
    #     for v_idx in range(len(v_lis)):
    #         if len(v_lis[v_idx]) < max_state:
    #             v_lis[v_idx] = np.append(v_lis[v_idx], [np.max(v_lis[v_idx]) for i in range(max_state - len(v_lis[v_idx]))])
    
    if m > idx:
        print("Multiple assginment.")
        multi = True

   

    return nodes, edges, set(cliques), states, unary_energy, pair_energy, max_state, multi, zero_e