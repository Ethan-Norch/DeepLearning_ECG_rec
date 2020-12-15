import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['CUDA_VISIBLE_DIVICES']='0,1,2,3'
# 调用GPU，普通电脑运行不了

RATIO = 0.1
PATH = 'data/心电智能大赛/preliminary_cl/'
def loaddata():
    with open(PATH + 'reference.txt', 'r') as f:
        label = []
        for i in f.readlines():
            i = i.strip('\n')
            label.append(float(i.replace('\t','')[-1]))
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
def ECGnet():
    def block3(x, K, filter_shape):
        y = BatchNormalization()(x)
        y = Activation("relu")(y)
        y = Conv1D(K, 1, strides=2)(y)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(K, filter_shape, strides=2, padding="same")(x)
        for i in range(2):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(K, filter_shape, strides=1, padding="same")(x)
        z = GlobalMaxPooling1D()(x)
        z = Dense(4, activation="relu")(z)
        z = Dense(K, activation="sigmoid")(z)
        x = Multiply()([x, z])
        y = Add()([x, y])
        for i in range(3):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(K, filter_shape, strides=1, padding="same")(x)
        z = GlobalMaxPooling1D()(x)
        z = Dense(16, activation="relu")(z)
        z = Dense(K, activation="sigmoid")(z)
        x = Multiply()([x, z])
        return Add()([x, y])
    def block2(x, K, filter_shape):
        for i in range(3):
            y = BatchNormalization()(x)
            y = Activation("relu")(y)
            y = Conv2D(K, filter_shape, padding="same")(y)
        z = GlobalMaxPooling2D()(y)
        z = Dense(8, activation="relu")(z)
        z = Dense(32, activation="sigmoid")(z)
        y = Multiply()([y,z])
        return Add()([x, y])
    def block1(x, K, filter_shape):
        # SE ResNet
        y = Conv2D(K, (1,1), strides=(2,1))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(K,filter_shape,strides=(2,1), padding="same")(x)
        z = GlobalMaxPooling2D()(x)
        z = Dense(8, activation="relu")(z)
        z = Dense(32, activation="sigmoid")(z)
        x = Multiply()([x, z])
        return Add()([x,y])

    def scale(x, filter_shape1, filter_shape2):
        for i in range(3):
            x = block2(x, 32, filter_shape1)
        x = Reshape([x.shape[1],x.shape[2]*x.shape[3]])(x)
        for i in range(4):
            x = block3(x, x.shape[2], filter_shape2)
        x = GlobalAveragePooling1D()(x)
        return x

    input = Input(shape=(5000,12,1))
    x = Conv2D(32, (50,1), strides=(2,1))(input)
    x = block1(x, 32, (15, 1))
    x = block1(x, 32, (15, 1))
    x = block1(x, 32, (15, 1))
    a = scale(x, (3,1), 3)
    b = scale(x, (5,1), 5)
    c = scale(x, (7,1), 7)
    x = Concatenate()([a,b,c])
    x = Dropout(0.5)(x)
    x = Dense(2, activation="sigmoid")(x)
    return Model(inputs=input, outputs=x)

model = ECGnet()
model.summary()
#
X_train, Y_train, X_test, Y_test = loaddata()
score = []
for lra in [0.01]:
    for batchs in [128]:
        for mom in [0.3]:
            model = ECGnet()
            model.compile(optimizer=tf.optimizers.SGD(tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps = 60, decay_rate=0.96), momentum=mom),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
# 定义TensorBoard对象
            checkpoint_path="./checkpoint"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1)
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./log")
# # 训练与验证
            history = model.fit(X_train, Y_train, epochs=200,
                    batch_size=batchs,
                    validation_split=0.15)
                    # callbacks=[tensorboard])
            print(history.history)
            fig = plt.figure()
            plt.plot(history.history['accuracy'], label='training acc')
            plt.plot(history.history['val_accuracy'], label='val acc')
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(loc='lower right')
            plt.show()
            fig = plt.figure()
            plt.plot(history.history['loss'], label='training loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.show()
            tf.keras.models.save_model(model, "model/"+"ECGNet1.pb")
            print("the score with initial lr "+str(lra)+" is",model.evaluate(X_test, Y_test, verbose=0))

