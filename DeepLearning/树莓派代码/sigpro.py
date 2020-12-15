'''
实时心电信号处理
'''

import pywt
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("ecgdata500-2.txt")
coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
threshold = (np.median(np.abs(cD1))/0.6745)*(np.sqrt(2*np.log(len(cD1))))
cD1.fill(0)
cD2.fill(0)
for i in range(1, len(coeffs)-2):
    coeffs[i]=pywt.threshold(coeffs[i], threshold)
rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
plt.figure(figsize=(20,6))
plt.subplot(3,1,1)
plt.plot(data)
plt.title("raw data")
plt.subplot(3,1,2)
plt.plot(rdata)
plt.title("new data")
plt.show()