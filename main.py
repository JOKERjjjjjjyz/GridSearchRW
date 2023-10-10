import time

import dataloader
import world
import torch
from dataloader import Loader
import sys
import scipy.sparse as sp
from train import *
import numpy as np
from scipy.sparse import csr_matrix
import torch.sparse
if __name__ == '__main__':
    if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        dataset = dataloader.Loader(path="./data/"+world.dataset)
    elif world.dataset == 'lastfm':
        dataset = dataloader.Loader(path="./data")
    core = int(world.CORES)
    graph,norm_graph = dataset.getSparseGraph()
    C=norm_graph
    print(graph)
    print("haha")
    print(C)

    print(type(graph),type(C))
    M = dataset.n_users
    N = dataset.m_items
    print(M,N)
    unit_matrix = sp.identity(M+N, format='csr')
    K_value = eval(world.topks)
    K = K_value[0]
    alpha = world.config['lr']
    vector_propagate = np.zeros((M + N, N))
    print(vector_propagate.shape)
    testarray = [[] for _ in range(M)]
    uservector = dataset.UserItemNet
    print(type(uservector))
    for idx, user in enumerate(dataset.test):
        testarray[idx] = dataset.test[user]

    with open(f"{world.dataset}_{alpha}_{K}_recall.txt", 'w') as file:
        # 在需要时写入内容
        file.write(f"This is {world.dataset}_{alpha}_{K}_recall:\n")
    for i in range(2,K+1):
        print("epoch",i,"start here")
        C = C.dot(norm_graph) * (1-alpha) + alpha * unit_matrix
        C_user = Mrow(C,M)
        vector_propagate = C_user.dot(uservector)
        print("epoch",i," finished")
        recall = Ktop(uservector, rowM(vector_propagate,M), M, N, 20,testarray)
        recall = recall / dataset.testDataSize
        with open(f"{world.dataset}_{alpha}_{K}_20_recall.txt", 'a') as file:
            file.write(f"epoch:{i} : topk50 ver:  recall: {recall}\n")