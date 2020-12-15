import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import os
'''
项目为单导联
'''
# project / data /data
#                 datapro.py
#         /checkpoint / epoch_loss
#         /figure / tensorboard
#         /model / train.py predict.py test.py   my_resnet.py ...

# 0.1 * batchsize/256
# 标签平滑
# 提前停止

# os.environ['CUDA_VISIBLE_DIVICES']='0'  调用GPU
RATIO = 0.1
PATH = 'dataset/心电智能大赛/preliminary_cl/'
def loaddata():
    with open(PATH + 'reference.txt', 'r') as f:
        label = []
        for i in f.readlines():
            i = i.strip('\n')
            label.append(float(i.replace('\t', '')[-1]))
        label = np.array(label)

    data = []
    for i in range(101,701):
        file = sio.loadmat(PATH+'TRAIN/TRAIN'+str(i)+'.mat')['data'][0].reshape(-1,1)
        file = MinMaxScaler(feature_range=(0, 1)).fit_transform(file)
        if i ==101:
            data = file
        elif i == 102:
            data = np.array([data,file])
            data = data.tolist()
        else:
            data.append(file.tolist())
    data = np.array(data)
    print(data.shape)
    data = data.reshape((600,5000))
    train = np.hstack((data,label.reshape(-1,1)))
    train = train.repeat(2,axis=0)
    np.random.shuffle(train)
    # 数据集及其标签集
    X = train[:, :5000].reshape(-1, 5000, 1)
    Y = train[:, 5000]
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train.reshape(-1,5000,1,1), Y_train.reshape(-1,1), X_test.reshape(-1,5000,1,1), Y_test.reshape(-1,1)
def ECGnet():
    def block3(x, K, filter_shape):
        y = BatchNormalization()(x)
        y = Activation("relu")(y)
        y = Conv1D(K, 1, strides=2, padding="same")(y)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(K, filter_shape, strides=2, padding="same")(x)
        for i in range(2):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(K, filter_shape, strides=1, padding="same")(x)
        z = GlobalMaxPooling1D()(x)
        z = Dense(4, activation="relu")(z)
        z = Dense(32, activation="sigmoid")(z)
        x = Multiply()([x, z])
        y = Add()([x, y])
        for i in range(3):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(K, filter_shape, strides=1, padding="same")(x)
        z = GlobalMaxPooling1D()(x)
        z = Dense(4, activation="relu")(z)
        z = Dense(32, activation="sigmoid")(z)
        x = Multiply()([x, z])
        return Add()([x, y])
    def block2(x, K, filter_shape):
        y = x
        for i in range(3):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(K, filter_shape, padding="same")(x)
        a = x
        x = GlobalMaxPooling2D()(x)
        x = Dense(8, activation="relu")(x)
        x = Dense(32, activation="sigmoid")(x)
        x = Multiply()([a,x])
        return Add()([x, y])
    def block1(x, K, filter_shape):
        # SE ResNet
        y = Conv2D(K, (1,1), strides=(2,1))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(K,filter_shape,strides=(2,1), padding="same")(x)
        a = x
        x = GlobalMaxPooling2D()(x)
        x = Dense(8, activation="relu")(x)
        x = Dense(32, activation="sigmoid")(x)
        x = Multiply()([a, x])
        return Add()([x,y])

    def scale(x, filter_shape1, filter_shape2):
        for i in range(3):
            x = block2(x, 32, filter_shape1)
        x = Reshape([x.shape[1],x.shape[2]*x.shape[3]])(x)
        print(x)
        for i in range(4):
            x = block3(x, 32, filter_shape2)
        x = GlobalAveragePooling1D()(x)
        return x

    input = Input(shape=(5000,1,1))
    x = Conv2D(32, (50,1), strides=(2,1))(input)
    x = block1(x, 32, (15, 1))
    x = block1(x, 32, (15, 1))
    x = block1(x, 32, (15, 1))
    a = scale(x, (3,1), 3)
    b = scale(x, (5,1), 5)
    c = scale(x, (7,1), 7)
    x = Concatenate()([a,b,c])
    x = Dense(2, activation="sigmoid")(x)
    return Model(inputs=input, outputs=x)
model = ECGnet()
# tf.keras.utils.plot_model(model, to_file='ECG Model.png', show_shapes=True, show_layer_names=True) 查看模型
model.summary()
X_train, Y_train, X_test, Y_test = loaddata()
model.compile(optimizer=tf.optimizers.SGD(lr = 0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 定义TensorBoard对象
logdir = os.path.join("./log")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# 训练与验证
history = model.fit(X_train, Y_train, epochs=20,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[tensorboard])
# print(X_test.shape)
# print(Y_train)
# print(model.predict(X_test))