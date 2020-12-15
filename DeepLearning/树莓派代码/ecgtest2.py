'''
此代码用于测试实时心率是否有问题
'''
import numpy as np
print("="*30)
print("loading model now ......")
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='./model.tflite')

import time
import Adafruit_ADS1x15 as ada
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pywt
print("="*30)
print("starting to collect your ecg data now.\nPlease remain calm...")
adc = ada.ADS1115()
adc.start_adc(3, 2/3, 475)
data=[]
while len(data)<5000:
    data.append(adc.get_last_result())
data=np.array(data)
data = data.reshape((5000,1))
data=MinMaxScaler(feature_range=(0,1)).fit_transform(data)
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
plt.show()

print("="*30)
print("analysing your ecgdata using ECGNet")
tt = np.array(rdata)
interpreter.allocate_tensors()
inputIndex=interpreter.get_input_details()[0]["index"]
outputIndex = interpreter.get_output_details()[0]["index"]
tt = tt.reshape((-1,5000,1,1))
tt = tt.astype(np.float32)
interpreter.set_tensor(inputIndex, tt)
interpreter.invoke()
prediction = interpreter.get_tensor(outputIndex)[0]
if prediction[0]>prediction[1]:
    print("there's something wrong with your ecgdata.\nThe suggest for you is to check your heart again in a hospital")
else:
    print("so far,you are healthy. keep exercising and stay fit.") 
