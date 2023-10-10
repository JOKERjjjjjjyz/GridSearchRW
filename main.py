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
    K = 7
    with open(f"{world.dataset}_layer{K}_top20_recall.txt", 'w') as file:
        # 在需要时写入内容
        file.write(f"This is {world.dataset}_maxlayer{K}_top20_recall:\n")
    for ii in range(1,10):
        alpha = 0.1 *ii
        with open(f"{world.dataset}_layer{K}_top20_recall.txt", 'a') as file:
            file.write(f"alpha={alpha} :\n")
        core = int(world.CORES)
        graph,norm_graph = dataset.getSparseGraph()
        C=norm_graph
        M = dataset.n_users
        N = dataset.m_items
        unit_matrix = sp.identity(M+N, format='csr')
        # K_value = eval(world.topks)
        # K = K_value[0]
        # alpha = world.config['lr']
        vector_propagate = np.zeros((M + N, N))
        print(vector_propagate.shape)
        testarray = [[] for _ in range(M)]
        uservector = dataset.UserItemNet
        print(type(uservector))
        for idx, user in enumerate(dataset.test):
            testarray[idx] = dataset.test[user]
        for i in range(2,K+1):
            print("epoch",i,"start here")
            # C = C.dot(norm_graph) * (1-alpha) + alpha * unit_matrix
            C = C.dot(norm_graph) * (1 - alpha) + alpha * unit_matrix
            C_user = Mrow(C,M)
            vector_propagate = C_user.dot(uservector)
            print("epoch",i," finished")
            recall = Ktop(uservector, rowM(vector_propagate,M), M, N, 20,testarray)
            recall = recall / dataset.testDataSize
            with open(f"{world.dataset}_layer{K}_top20_recall.txt", 'a') as file:
                file.write(f"layer:{i} : topk20 ver:  recall: {recall}\n")