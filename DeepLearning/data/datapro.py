# 心电大赛数据处理

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio

RATIO = 0.1
PATH = '心电智能大赛/preliminary_cl/'
def loaddata():
    with open(PATH + 'reference.txt', 'r') as f:
        label = []
        for i in f.readlines():
            i = i.strip('\n')
            label.append(float(i.replace('\t', '')[-1]))
        label = np.array(label)

    data = []
    for i in range(101,701):
        file = sio.loadmat(PATH+'TRAIN/TRAIN'+str(i)+'.mat')['data']
        file = MinMaxScaler(feature_range=(0, 1)).fit_transform(file)
        if i ==101:
            data = file
        elif i == 102:
            data = np.array([data,file])
            data = data.tolist()
        else:
            data.append(file.tolist())
    data = np.array(data)
    data = data.reshape((-1,60000))
    data = np.hstack((data,label.reshape(-1,1)))
    data = data.repeat(3,axis=0)
    np.random.shuffle(data)
    # 数据集及其标签集
    X = data[:, :60000].reshape(-1, 5000, 12)
    Y = data[:, 60000]
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train.reshape(-1,5000,12,1), Y_train.reshape(-1,1), X_test.reshape(-1,5000,12,1), Y_test.reshape(-1,1)
X_train, Y_train, X_test, Y_test = loaddata()

