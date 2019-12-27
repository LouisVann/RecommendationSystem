import numpy as np
import pandas as pd
import os

# global variables
file_path = 'train.csv'
target_path = 'test_index.csv'
load_path = 'user_similarity_matrix.txt'
X = None  # sparse matrix of ratings
user_similarity_matrix = None
rating_mean = 0

# hyper-parameters
k_num = 30  # num of neighbors

'''
k   rmse
6   0.9917
10  0.9673
20  0.9479
30  0.9435
'''


# get raw tuples from the given file
def get_raw_data(path):
    df = pd.read_csv(path)
    user_num = df[['userID']].max()[0] + 1
    item_num = df[['itemID']].max()[0] + 1

    train_data = df.sample(frac=0.8, random_state=0)[['userID', 'itemID', 'rating']]
    test_data = df.drop(train_data.index)[['userID', 'itemID', 'rating']]

    # # 为了out.csv效果更好
    # train_data = df[['userID', 'itemID', 'rating']]

    return user_num, item_num, train_data.values, test_data.values


# dense matrix to sparse matrix
def get_rating_matrix(user_num, item_num, train_data):
    matrix = np.zeros(shape=[user_num, item_num])
    for tup in train_data:
        matrix[int(tup[0]), int(tup[1])] = tup[2]
    return matrix


def _find_common_index(a, b):
    return np.where((a > 0) & (b > 0))


def _calc_cos_similarity(a, b):
    a_len = np.linalg.norm(a)
    b_len = np.linalg.norm(b)
    if a_len == 0 or b_len == 0:
        return 0
    else:
        return np.dot(a, b) / (a_len * b_len)


# ignore the entries that are not rated
def get_rating_mean(a):
    assert rating_mean != 0
    entries_used = a[np.where(a > 0)]
    if len(entries_used) == 0:
        return rating_mean
    else:
        return np.mean(entries_used)


# @param: a and b are both user rating vectors
# @return: user similarity ranging from -1 to 1
def calc_user_similarity(a, b):
    assert len(a.shape) == 1 and len(b.shape) == 1 and a.shape[0] == b.shape[0]
    com_ids = _find_common_index(a, b)
    if len(com_ids) == 0:
        return 0
    a1 = a[com_ids] - get_rating_mean(a)
    b1 = b[com_ids] - get_rating_mean(b)
    return _calc_cos_similarity(a1, b1)


def calc_rmse(a, b):
    assert len(a.shape) == 1 and len(b.shape) == 1 and a.shape[0] == b.shape[0]
    rmse = np.sqrt(np.sum(np.square(a - b)) / b.size)
    return rmse


def get_user_similarity_matrix():
    mat = np.zeros(shape=[X.shape[0], X.shape[0]])
    for i in range(X.shape[0]):
        if i % 200 == 0:
            print('get_user_similarity_matrix: %d user finished' % i)
        for j in range(X.shape[0]):
            if mat[i, j] == 0:
                sim = calc_user_similarity(X[i], X[j])
                mat[i, j] = sim
                mat[j, i] = sim
    return mat


# @param: u is the userID of the current user
# @param: i is the itemID of the target movie
# purpose: get the neighbors of the u-th user
def get_neighbors(u, iid):
    ids = []
    sorted = np.argsort(user_similarity_matrix[u])
    for i in range(0, len(user_similarity_matrix[u])):
        this_uid = sorted[len(user_similarity_matrix[u]) - i - 1]
        if this_uid != u and X[this_uid, iid] != 0:
            ids.append(this_uid)
        if len(ids) >= k_num:
            break
    return np.array(ids)


def predict(uid, iid):
    rating = get_rating_mean(X[uid])
    nids = get_neighbors(uid, iid)
    s1, s2 = 0, 0
    for nid in nids:
        s1 += user_similarity_matrix[uid, nid] * (X[nid, iid] - get_rating_mean(X[nid]))
        s2 += abs(user_similarity_matrix[uid, nid])
    if s2 != 0:
        rating += s1 / s2

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
        user_similarity_matrix = np.loadtxt(load_path)
    else:
        user_similarity_matrix = get_user_similarity_matrix()
        np.savetxt(load_path, user_similarity_matrix)
    print('user similarity matrix constructed')

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
    output_df.to_csv('out_1.csv', index=None)
