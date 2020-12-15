import numpy as np
from sklearn.preprocessing import MinMaxScaler
print("="*30)
print("loading model now ......")
# import tensorflow as tf
# interpreter = tf.lite.Interpreter(model_path='./model.tflite')

import time
# import Adafruit_ADS1x15 as ada
import matplotlib.pyplot as plt
import pywt
print("="*30)
print("starting to collect your ecg data now.\nPlease remain calm...")
time.sleep(12)
data=np.loadtxt("D:\\anaconda3\\envs\\myTensorflow\\ECG\\Tang\\ecgdata500-2.txt")
print("="*30)
print("collecting ecgdata finished, now processing your ecgdata......")
coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
threshold = (np.median(np.abs(cD1))/0.6745)*(np.sqrt(2*np.log(len(cD1))))
cD1.fill(0)
cD2.fill(0)
cD3.fill(0)
for i in range(1, len(coeffs)-3):
    coeffs[i]=pywt.threshold(coeffs[i], threshold)
rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
print("="*30)
print("showing your ecgdata")
plt.figure(figsize=(20,4))
plt.subplot(3,1,1)
plt.plot(data)
plt.title("raw data")
plt.subplot(3,1,2)
plt.plot(rdata)
plt.title("new data")
plt.savefig('D:\\anaconda3\\envs\\myTensorflow\\ECG\\Tang\\ecgtest3.png')
plt.show()

print("="*30)
print("analysing your ecgdata using ECGNet")
tt = np.array(rdata).reshape((5000,1))
tt = MinMaxScaler(feature_range=(0,1)).fit_transform(tt)
interpreter.allocate_tensors()
inputIndex=interpreter.get_input_details()[0]["index"]
outputIndex = interpreter.get_output_details()[0]["index"]
tt = tt.reshape((-1,5000,1,1))
tt = tt.astype(np.float32)
interpreter.set_tensor(inputIndex, tt)
interpreter.invoke()
prediction = interpreter.get_tensor(outputIndex)[0]
print(prediction)
print("so far,you are healthy. keep exercising and stay fit.")
