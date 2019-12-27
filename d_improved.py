import numpy as np
import pandas as pd
from a_user_based import get_raw_data, calc_rmse

# global variables
file_path = 'train.csv'
target_path = 'test_index.csv'

# hyper-parameters
f = 30  # num of features
learning_rate = 0.01
regularizer_rate = 0.095  # 0.1
batch_size = 50
epoch_num = 35

'''
f   lr      rr  e   rmse
6   0.001   0.5 10  0.9517
6   0.001   0.2 10  0.9434
10  0.001   0.2 10  0.9446
4   0.001   0.2 10  0.9433
10  0.001   0.2 20  0.9240
15  0.001   0.2 25  0.9174
15  0.01    0.2 25  0.8968 ***
32  0.01    0.2 25  0.8964 ***
32  0.01    0.2 40  0.8965
32  0.01    0.3 25  0.9003
16  0.01    0.18 25 0.8963 ***
16  0.02    0.18 25 0.8969
16  0.005   0.18 25 0.8970
16  0.01    0.1  25 0.8916 ***
28  0.01    0.1 25  0.8898
32  0.01    0.1 25  0.8897 ***
50  0.01    0.1 25  0.891
40  0.01    0.1 25  0.8901
32  0.01    0.1 25  0.8891 ***
batch_size=32 0.8886
batch_size=50 0.8874
0.15 0.72
0.08 0.666
0.06 0.58998 X
'''


class SVDPP_MF:
    def __init__(self, f=10, learning_rate=0.001, regularizer_rate=0.5, batch_size=50, epoch_num=10):
        self.f = f
        self.lr = learning_rate
        self.regularizer_rate = regularizer_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num

        self.n, self.m, self.train_data, self.test_data = get_raw_data(file_path)
        np.random.shuffle(self.train_data)

        self.P = np.random.normal(0.0, 0.1, [self.n, self.f])
        self.Q = np.random.normal(0.0, 0.1, [self.m, self.f])
        self.bu = np.random.normal(0.0, 0.1, [self.n, ])
        self.bi = np.random.normal(0.0, 0.1, [self.m, ])
        self.b_avg = np.mean(self.train_data[:, 2])

        print('model initialized.')

    def train(self):  # Mini-Batch Gradient Descent, MBGD
        for e in range(self.epoch_num):
            print('training begin: ', e, ' epoch')
            # np.random.shuffle(self.train_data)

            batch_num = self.train_data.shape[0] // self.batch_size + 1
            for batch_id in range(batch_num):
                train_data = self.train_data[batch_id * batch_size : (batch_id + 1) * batch_size] if batch_id < batch_num \
                    else self.train_data[batch_id * batch_size : self.train_data.shape[0]]
                uids, iids, ratings = train_data[:, 0].astype(np.int32), train_data[:, 1].astype(np.int32), train_data[:, 2]
                user_latents = self.P[uids]
                item_latents = self.Q[iids]
                user_biases = self.bu[uids]
                item_biases = self.bi[iids]
                predictions = np.sum(np.multiply(user_latents, item_latents), axis=1) + user_biases + item_biases + self.b_avg
                errs = ratings - predictions

                for i in range(train_data.shape[0]):
                    uid, iid, _ = train_data[i, :].astype(np.int32)
                    self.P[uid] += self.lr * errs[i] * item_latents[i] - self.lr * self.regularizer_rate * user_latents[i]
                    self.Q[iid] += self.lr * errs[i] * user_latents[i] - self.lr * self.regularizer_rate * item_latents[i]
                    self.bu[uid] += self.lr * errs[i] - self.lr * self.regularizer_rate * self.bu[uid]
                    self.bi[iid] += self.lr * errs[i] - self.lr * self.regularizer_rate * self.bi[iid]

            # # Stochastic Gradient Descent，SGD
            # for tup in self.train_data:
            #     uid, iid, rating = tup
            #     uid, iid = int(uid), int(iid)
            #     err = rating - self._predict(uid, iid)
            #     self.P[uid] += self.lr * err * self.Q[iid] - self.lr * self.regularizer_rate * self.P[uid]
            #     self.Q[iid] += self.lr * err * self.P[uid] - self.lr * self.regularizer_rate * self.Q[iid]
            #     self.bu[uid] += self.lr * err - self.lr * self.regularizer_rate * self.bu[uid]
            #     self.bi[iid] += self.lr * err - self.lr * self.regularizer_rate * self.bi[iid]

    def _predict(self, uid, iid):
        rating = np.dot(self.P[uid], self.Q[iid])\
                 + self.bu[uid] + self.bi[iid] + self.b_avg
        return rating

    def predict_batch(self, input_data):
        input_data = input_data[:, 0:2].astype(np.int32)
        uids, iids = input_data[:, 0], input_data[:, 1]
        ratings = np.sum(np.multiply(self.P[uids], self.Q[iids]), axis=1)\
                  + self.bu[uids] + self.bi[iids] + self.b_avg

        ratings[np.where(ratings < 0.5)] = 0.5
        ratings[np.where(ratings > 5.0)] = 5.0

        return ratings


if __name__ == '__main__':
    model = SVDPP_MF(f=f, learning_rate=learning_rate, regularizer_rate=regularizer_rate, batch_size=batch_size, epoch_num=epoch_num)
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
    output_df.to_csv('out_4.csv', index=None)

    another = pd.DataFrame(predicts, columns=['rating'])
    another.to_csv('another.csv')

