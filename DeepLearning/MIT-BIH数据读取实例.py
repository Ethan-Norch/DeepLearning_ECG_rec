import wfdb
import numpy as np
import pywt
import matplotlib.pyplot as plt
record = wfdb.rdrecord('./data/100/100', channel_names=['MLII'])
data = record.p_signal.flatten()[100:1600]

# 用db5作为小波基，对心电数据进行9尺度小波变换
coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
# 使用软阈值滤波
threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
# 将高频信号cD1、cD2置零
cD1.fill(0)
cD2.fill(0)
# 将其他中低频信号按软阈值公式滤波
for i in range(1, len(coeffs) - 2):
	coeffs[i] = pywt.threshold(coeffs[i], threshold)
rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
plt.figure(figsize=(10, 4))
plt.subplot(2,1,1)
plt.plot(data)
plt.subplot(2,1,2)
plt.plot(rdata)
plt.show()