import numpy as np
import pandas as pd
import os
from a_user_based import get_raw_data, get_rating_matrix, _find_common_index, _calc_cos_similarity, calc_rmse

# global variables
file_path = 'train.csv'
target_path = 'test_index.csv'
load_path = 'item_similarity_matrix.txt'
X = None  # sparse matrix of ratings
item_similarity_matrix = None
rating_mean = 0

# hyper-parameters
k_num = 50  # num of neighbors

'''
k   rmse
5   1.078
10  1.033
20  1.009
30  1.001
50  0.9984
100 0.9988
'''


# ignore the entries that are not rated
def get_rating_mean(vector):
    assert rating_mean != 0
    entries_used = vector[np.where(vector > 0)]
    if len(entries_used) == 0:
        return rating_mean
    else:
        return np.mean(entries_used)


# @param: a and b are both item rating vectors
# @return: item similarity ranging from -1 to 1
def calc_item_similarity(a, b):
    assert len(a.shape) == 1 and len(b.shape) == 1 and a.shape[0] == b.shape[0]
    com_ids = _find_common_index(a, b)
    if len(com_ids) == 0:
        return 0
    a1 = a[com_ids]
    b1 = b[com_ids]
    return _calc_cos_similarity(a1, b1)


def get_item_similarity_matrix():
    mat = np.zeros(shape=[X.shape[1], X.shape[1]])
    for i in range(X.shape[1]):
        if i % 200 == 0:
            print('get_item_similarity_matrix: %d item finished' % i)
        for j in range(X.shape[1]):
            if mat[i, j] == 0:
                sim = calc_item_similarity(X[:, i], X[:, j])
                mat[i, j] = sim
                mat[j, i] = sim
    return mat


# @param: iid is the itemID of the current movie
# @param: u is the userID of the target user
# purpose: get the neighbors of the iid-th item
def get_neighbors(iid, u):
    ids = []
    sorted = np.argsort(item_similarity_matrix[iid])
    for i in range(0, len(item_similarity_matrix[iid])):
        this_iid = sorted[len(item_similarity_matrix[iid]) - i - 1]
        if this_iid != iid and X[u, this_iid] != 0:
            ids.append(this_iid)
        if len(ids) >= k_num:
            break
    return np.array(ids)


def predict(uid, iid):
    rating = 0
    nids = get_neighbors(iid, uid)
    s1, s2 = 0, 0
    for nid in nids:
        s1 += item_similarity_matrix[iid, nid] * X[uid, nid]
        s2 += abs(item_similarity_matrix[iid, nid])
    if s2 != 0:
        rating += s1 / s2
    else:
        rating = get_rating_mean(X[:, iid])

    if rating < 0.5:
        rating = 0.5
    if rating > 5.0:
        rating = 5.0

    return rating


if __name__ == '__main__':
    user_num, item_num, train_data, test_data = get_raw_data(file_path)
    X = get_rating_matrix(user_num, item_num, train_data)
    rating_mean = np.mean(train_data[:, 2])

    if os.path.exists(load_path):
        item_similarity_matrix = np.loadtxt(load_path)
    else:
        item_similarity_matrix = get_item_similarity_matrix()
        np.savetxt(load_path, item_similarity_matrix)
    print('item similarity matrix constructed')

    # predict part #
    predicts = np.zeros(test_data.shape[0])
    for i in range(test_data.shape[0]):
        tup = test_data[i]
        predicts[i] = predict(int(tup[0]), int(tup[1]))
    rmse = calc_rmse(predicts, test_data[:, 2])
    print('rmse on test_data: ', rmse)

    # final prediction #
    df = pd.read_csv(target_path)
    target_data = df.values  # (userID, itemID)
    predicts = np.zeros(target_data.shape[0])
    for i in range(target_data.shape[0]):
        tup = target_data[i]
        predicts[i] = predict(int(tup[0]), int(tup[1]))
    output_df = pd.DataFrame(target_data, columns=['userID', 'itemID'])
    output_df.insert(2, 'rating', predicts)
    output_df.to_csv('out_2.csv', index=None)
