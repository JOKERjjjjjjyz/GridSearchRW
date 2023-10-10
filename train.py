import numpy as np
from multiprocessing import Pool
import math
def rowM(matrix,M):
    B = matrix[:M]
    return B
def Mrow(matrix,M):
    B = matrix[:M]
    B = B.transpose()
    print (B.shape)
    return B

def Ktop(vector_origin,vector_propagate,M,N,k,test):
    recall = 0
    count = 0
    vector = vector_propagate - vector_origin
    vector_array = vector.toarray()
    topk_indices = np.argsort(vector_array, axis=1)[:, -k:]
    print (topk_indices,type(topk_indices),topk_indices.shape)
    for user in range(M):
        for item in test[user]:
            count+=1
            print("Ktop:count")
            if item in topk_indices[user]: recall +=1
    return recall
    # for user in range(M):
    #     print("topK of user",user)
    #     sorted_indices = np.argsort(vector[user])
    #     topk_indices = sorted_indices[-k:]
    #     for idx in topk_indices:
    #         recommend_vector[user][idx] = 1
    #         recommendList.append((user,idx))
    #     print("user",user,"finished")
    # return recommendList, recommend_vector
def topK(vector, M, N, k, user_start, user_end):
    recommendList = []
    # recommend_vector = [np.zeros(N) for _ in range(M)]
    for user in range(user_start, user_end):
        dense = vector[user].toarray()
        dense_vector_user = dense[0]
        # print("user",user,":",dense_vector_user)
        sorted_indices = np.argsort(dense_vector_user)
        # print("user", user, ":", sorted_indices)
        topk_indices = sorted_indices[-k:]
        # print("user", user, ":", topk_indices)
        for idx in topk_indices:
            # print(user,idx)
            # recommend_vector[user][idx] = 1
            recommendList.append((user, idx))
    return recommendList

def parallel_topK(vector_origin, vector_propagate, M, N, k, num_cores):
    chunk_size = M // num_cores
    vector = vector_propagate - vector_origin
    # print(vector)
    pool = Pool(num_cores)
    results = []
    for i in range(num_cores):
        user_start = i * chunk_size
        user_end = (i + 1) * chunk_size if i < num_cores - 1 else M
        results.append(pool.apply_async(topK, (vector, M, N, k, user_start, user_end)))
    pool.close()
    pool.join()

    recommendList = []
    recommend_vector = [np.zeros(N) for _ in range(M)]
    for result in results:
        partial_recommendList = result.get()
        recommendList.extend(partial_recommendList)
        # for user in range(M):
        #     recommend_vector[user] += partial_recommend_vector[user]
    return recommendList

def evaluate(recommendList, test):
    count = 0
    count2 = 0
    print("Evaluating...")
    RecLenth = len(recommendList)
    for tuple_item in recommendList:
        count2 +=1
        user = tuple_item[0]
        item = tuple_item[1]
        # testnp = numpy_array = np.array(test)
        for test_item in test[user]:
            if (test_item == item):
                count += 1
                break
    return count

# def Ktop(vector_origin,vector_propagate,M,N,k):
#     recommendList = []
#     recommend_vector = [np.zeros(N) for _ in range(M)]
#     vector = vector_propagate - 1000*vector_origin
#     print(type(vector_origin),vector_origin.shape,type(vector_propagate),vector_propagate.shape)
#     for user in range(M):
#         print("topK of user",user)
#         sorted_indices = np.argsort(vector[user])
#         topk_indices = sorted_indices[-k:]
#         for idx in topk_indices:
#             recommend_vector[user][idx] = 1
#             recommendList.append((user,idx))
#         print("user",user,"finished")
#     return recommendList, recommend_vector