'''
最简单ADS1115例子
每0.5s输出一次四个输入口的数据
'''
import time
import Adafruit_ADS1x15 as ada

adc = ada.ADS1115()

while True:
    value=[0]*4
    for i in range(4):
        value[i] = adc.read_adc(i, 1)
    print('|{0:>6}|{1:>6}|{2:>6}|{3:>6}|'.format(*value))
    time.sleep(0.5)