'''
树莓派代码，用于ADS1115模型通信树莓派，连续读取数字信号
读取心电信号并进行预处理
'''
import Adafruit_ADS1x15 as ada
import matplotlib.pyplot as plt
import numpy as np
import pywt
adc = ada.ADS1115()
adc.start_adc(3, 2/3, 475)
'''
3是通道数
2/3是信号增益
475是采样率
'''
data=[]
while len(data)<5000:
    data.append(adc.get_last_result())
data=np.array(data)
coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
threshold = (np.median(np.abs(cD1))/0.6745)*(np.sqrt(2*np.log(len(cD1))))
cD1.fill(0)
cD2.fill(0)
cD3.fill(0)
for i in range(1, len(coeffs)-3):
    coeffs[i]=pywt.threshold(coeffs[i], threshold)
rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
plt.figure(figsize=(20,4))
plt.subplot(3,1,1)
plt.plot(data)
plt.title("raw data")
plt.subplot(3,1,2)
plt.plot(rdata)
plt.title("new data")
plt.show()
