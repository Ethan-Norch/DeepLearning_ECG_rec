# 该代码使用训练好的模型进行数据扩充，鉴于模型在验证集和测试集上的准确率都是100%
# 有理由相信该数据扩充是有效的

import numpy as np
# import tensorflow as tf
import scipy.io as sio
from tensorflow.keras.models import *
from sklearn.preprocessing import MinMaxScaler
testfilepath = "./data/心电智能大赛/preliminary_cl/TEST"
model = load_model('../model/ECGNet.pb')

def loaddata():
    data=[]
    for i in range(101,500):
        file = sio.loadmat(testfilepath+'/TEST'+str(i)+'.mat')['data']
        file = MinMaxScaler(feature_range=(0, 1)).fit_transform(file)
        if i ==101:
            data = file
        elif i == 102:
            data = np.array([data,file])
            data = data.tolist()
        else:
            data.append(file.tolist())
    data = np.array(data)
    data = data.reshape((-1,5000,12,1))
    return data
def lable():
    with open('心电智能大赛/preliminary_cl/reference.txt', 'r') as f:
        label = []
        for i in f.readlines():
            i = i.strip('\n')
            label.append(float(i.replace('\t','')[-1]))
        label = np.array(label)
        return label
lable = lable().tolist()
data = loaddata()
print(data.shape)
print(lable)
pre = model.predict(data)
for i in pre:
    if i[0]>i[1]:
        lable.append(0.0)
    else:
        lable.append(1.0)
print(lable)
lable = np.array(lable)
np.savetxt('./data/心电智能大赛/preliminary_cl/lable1.txt', lable)