from GG_net2 import *
import numpy as np
from time import time
import random
from itertools import chain, islice
from utils2 import *
import os
from tqdm import tqdm
import pandas as pd

path = "path to your data files"
files= os.listdir(path) 


data_loss = {}
data_res = {}

for file in files: 
    dims = [1024]
    r = len(dims)

    if True:

        nodes, edges, cliques, states, unary_energy_p, pair_energy_p, max_state, multi, zero_e = read_clique_model2(path+"/"+file)
  
        if zero_e:
            print("zero energy, skip")
            print("================================================================================================")
            print()
            continue
        print(file, len(nodes), len(cliques), len(edges))  
        
       
        masks = None

        nodes_num = len(nodes)

        for seed in [66, 666, 6666]:
        
            print("seed = ", seed)
            setup_seed(seed)
            for i in range(r):
                
                print("dim = ", dims[i])
                dim_embedding =  dims[i]
                gnn_hypers = {'num_features': dim_embedding, 'num_classes': max_state, "nerouns": 1024, 'dropout': 0.0}
                opt_params = {'lr': 0.0001}
                torch_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
                torch_device = torch.device( 'cpu')
                torch_dtype = torch.float
                num_iter = 100

                hnet = 1
                
                net, optimizer, embed = create_net(nodes_num, gnn_hypers, opt_params, torch_device, torch_dtype)
                

                masks = create_variable_class_mask(nodes_num, states, torch_device)
                net.mask = masks

                net.train()
                if hnet == 1:
                    res = train_hyper(nodes, edges, cliques, unary_energy_p, pair_energy_p, masks, net, 
                            num_iter, optimizer, embed, torch_device, dim_embedding)
                else:
                    res = train(nodes, edges, cliques, unary_energy_p, pair_energy_p, masks,net, 
                            num_iter, optimizer, embed, torch_device, dim_embedding)

                if not isinstance(res[1], bool):
                    check_feasi(res[1], states)
                else:
                    print("No result!")
                    break
                print()
    
        
        print("================================================================================================")
        print()

