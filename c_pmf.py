import numpy as np
import pandas as pd
from a_user_based import get_raw_data, get_rating_matrix, calc_rmse

# global variables
file_path = 'train.csv'
target_path = 'test_index.csv'

# hyper-parameters
f = 10  # num of features
regularizer_rate = 0.08  # 0.18
epoch_num = 30

'''
f       rmse rr=0.18 e=30
10      0.964
12      0.969
8       0.966

rr      rmse f=10 e=30
0.18    0.964
0.22    0.971
0.15    0.968

e       rmse f=10 rr=0.18
10      0.967
20      0.965
30      0.964
40      0.965

0.25  0.732
0.12  0.644
'''


class PMF:
    def __init__(self, f=10, regularizer_rate=0.5, epoch_num=10):
        self.f = f
        self.regularizer_rate = regularizer_rate
        self.epoch_num = epoch_num

        self.n, self.m, self.train_data, self.test_data = get_raw_data(file_path)
        self.R = get_rating_matrix(self.n, self.m, self.train_data)

        self.P = np.random.normal(0.0, 0.1, [self.n, self.f])
        self.Q = np.random.normal(0.0, 0.1, [self.m, self.f])
        print('model initialized.')

    def train(self):
        for e in range(self.epoch_num):
            print('training begin: ', e, ' epoch')

            # updating user latent factor matrix
            for uid in range(self.n):
                iid_list = np.where(self.R[uid] > 0)
                weight = len(iid_list[0]) if len(iid_list[0]) > 0 else 1
                p1 = self.regularizer_rate * np.eye(self.f) * weight + np.dot(self.Q[iid_list].T, self.Q[iid_list])  # shape: f * f
                p2 = np.sum(np.multiply(self.Q[iid_list], np.matrix([self.R[uid][iid_list] for i in range(self.f)]).T), axis=0)  # shape: 1 * f
                self.P[uid] = np.dot(np.linalg.inv(p1), p2.T).flatten()  # shape: f

            # updating item latent factor matrix
            for iid in range(self.m):
                uid_list = np.where(self.R[:, iid] > 0)
                weight = len(uid_list[0]) if len(uid_list[0]) > 0 else 1
                p1 = self.regularizer_rate * np.eye(self.f) * weight + np.dot(self.P[uid_list].T, self.P[uid_list])
                p2 = np.sum(np.multiply(self.P[uid_list], np.matrix([self.R.T[iid][uid_list] for i in range(self.f)]).T), axis=0)
                self.Q[iid] = np.dot(np.linalg.inv(p1), p2.T).flatten()

    def predict(self, uid, iid):
        rating = np.dot(self.P[uid], self.Q[iid])

        if rating < 0.5:
            rating = 0.5
        if rating > 5.0:
            rating = 5.0

        return rating

    def predict_batch(self, input_data):
        input_data = input_data[:, 0:2].astype(np.int32)
        ### way 1
        user_latents = self.P[input_data[:, 0]]
        item_latents = self.Q[input_data[:, 1]]
        ratings = np.sum(np.multiply(user_latents, item_latents), axis=1)

        # ### way 2
        # ratings = np.zeros(shape=input_data.shape[0])
        # for i in range(input_data.shape[0]):
        #     ratings[i] = self.predict(input_data[i, 0], input_data[i, 1])

        ratings[np.where(ratings < 0.5)] = 0.5
        ratings[np.where(ratings > 5.0)] = 5.0

        return ratings


if __name__ == '__main__':
    model = PMF(f=f, regularizer_rate=regularizer_rate, epoch_num=epoch_num)
    model.train()

    # predict part #
    test_data = model.test_data
    predicted = model.predict_batch(model.test_data)
    rmse = calc_rmse(predicted, test_data[:, 2])
    print('rmse on test_data: ', rmse)

    # final prediction #
    df = pd.read_csv(target_path)
    target_data = df.values  # (userID, itemID)
    predicts = model.predict_batch(target_data)

    output_df = pd.DataFrame(target_data, columns=['userID', 'itemID'])
    output_df.insert(2, 'rating', predicts)
    output_df.to_csv('out_3.csv', index=None)
