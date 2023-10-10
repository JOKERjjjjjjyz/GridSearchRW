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
    C_sum =C
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
    print(C_sum.shape)
    # vector_propagate = Mrow(C_sum,M).dot(uservector)
    # print("topK here")
    # recommendList, recommend_vector = topK(uservector, rowM(vector_propagate,M), M, N, 20)
    # count = evaluate(recommendList, testarray)
    # recall = count / dataset.testDataSize
    # print("sum ver:epoch:",1," recall:", recall)
    with open(f"{world.dataset}_{alpha}_{K}_recall.txt", 'w') as file:
        # 在需要时写入内容
        file.write(f"This is {world.dataset}_{alpha}_{K}_recall:\n")
    for i in range(2,K+1):
        print("epoch",i,"start here")
        # C = C.dot(norm_graph) * (alpha) * math.pow(1-alpha,i-1)
        C = C.dot(norm_graph) * (1-alpha) + alpha * unit_matrix
        # filename = f"{world.dataset}_matrix_{i}.npy"  # 文件名类似于 matrix_0.npy, matrix_1.npy, ...
        # np.save(filename, C)
        C_sum += C
        # filename = f"{world.dataset}_matrix_sum_{i}.npy"
        # np.save(filename, C_sum)
        C_user = Mrow(C,M)
        C_user_sum = Mrow(C_sum,M)
        vector_propagate = C_user.dot(uservector)
        # filename = f"{world.dataset}_vector_propagate_{i}.npy"
        # np.save(filename, vector_propagate)
        print("epoch",i," finished")
        # recommendList = parallel_topK(uservector, rowM(vector_propagate,M), M, N, 3,core)
        recall = Ktop(uservector, rowM(vector_propagate,M), M, N, 20,testarray)
        recall = recall / dataset.testDataSize
        # count = evaluate(recommendList, testarray)
        # recall = count / dataset.testDataSize
        with open(f"{world.dataset}_{alpha}_{K}_recall.txt", 'a') as file:
            file.write(f"epoch:{i} : not sum ver:  recall: {recall}\n")
        # print("not sum ver:epoch:",i," recall:", recall)
        # filename = f"{world.dataset}_recall_{i}.npy"
        # np.save(filename, recall)
        vector_propagate = C_user_sum.dot(uservector)
        s = time.time()
        recall = Ktop(uservector, rowM(vector_propagate,M), M, N, 20,testarray)
        e = time.time()
        u = e-s
        print("Ktop:",u,"s")
        recall = recall / dataset.testDataSize
        # s = time.time()
        # recommendList = parallel_topK(uservector, rowM(vector_propagate,M), M, N, 3,core)
        # filename = f"{world.dataset}_reclist_{i}.npy"
        # np.save(filename, recommendList)
        # recommend_vector_csr = csr_matrix(recommend_vector)
        # sp.save_npz(dataset.path + '/recommend_vector_{i}.npz', recommend_vector_csr)
        # count = evaluate(recommendList, testarray)
        # e = time.time()
        # u = e-s
        # print("topK:",u,"s")
        # recall = count / dataset.testDataSize
        with open(f"{world.dataset}_{alpha}_{K}_recall.txt", 'a') as file:
            file.write(f"epoch:{i} :     sum ver:  recall: {recall}\n")
        # print("sum ver:epoch:",i," recall:", recall)
        # filename = f"{world.dataset}_sum_recall_{i}.npy"
        # np.save(filename, recall)