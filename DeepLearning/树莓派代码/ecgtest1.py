'''
此代码用于树莓派上运行tflite模型
'''
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

tt = np.loadtxt('test.txt')
data = tt.reshape((5000,1))
tt=MinMaxScaler(feature_range=(0,1)).fit_transform(data)
interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()
inputIndex=interpreter.get_input_details()[0]["index"]
outputIndex = interpreter.get_output_details()[0]["index"]
tt = tt.reshape((-1,5000,1,1))
tt = tt.astype(np.float32)
interpreter.set_tensor(inputIndex, tt)
interpreter.invoke()
prediction = interpreter.get_tensor(outputIndex)
print(prediction)
