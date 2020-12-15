# 基于硬件树莓派0-Tensorflow的心电图识别



[TOC]



**项目流程**

```flow
st=>operation: 心电数据获取
op=>operation: 心电数据融合
cond=>operation: Tensorflow模型搭建（ECGNet）
sub1=>operation: 训练模型
io=>operation: 模型移植树莓派
a=>operation: 测试
st(right)->op(right)->cond(right)->sub1(right)->io(right)->a
```

> 本项目一共分为六大步骤。



## 1、心电数据获取

### 1.1心电数据

在一个心跳周期中，心脏受到外界刺激后会有规律的持续收缩并产生电激动，之后刺激消失后又会舒张，在这个过程中，会有大量的心肌细胞产生有规律的电位变化，通过人体表面的电极可以记录到电位变化的曲线，曲线经过放大也就是临床上的心电图，即心电信号（ECG）。心电信号是一种微弱生物电信号，有以下特征：

+ 微弱性：心电信号的幅值在10uV-5mⅤ范围，是低幅值信号。

+ 不稳定性：心电信号在不断地变化且容易受到环境干扰，覆盖大量噪声，导致心电信号的很多有价值信息被淹没，很难检测，且不同个体在不同时刻下的心电图都是不同的，即使同一个体在不同生理状态下波形也可能不同。

+ 低频性：心电信号的频率范围主要在0.05-100Hz内，主要能量集中分布在0.5-40Hz。一个完整的心拍主要由P波、QRS波群、T波、PR波段以及ST波段构成，不同波段分别反映了兴奋传导至心脏各部位的具体变化情况。PR间期和QT间期可以传递心电信号非常重要的生理信息，是心电信号中非常重要的特征。

  ![u=4056780702,4219508502&fm=26&gp=0](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/u=4056780702,4219508502&fm=26&gp=0.jpg)

​		[PhysioNet](https://www.physionet.org/about/database/)是一个提供生理信号记录的数据库（PhysioBank）和相关的开源软件（PhysioToolkit）的机构。PhysioBank拥有大量心电信号数据，大部分都是免费提供的，可以从网站上直接下载。

​		[MIT-BIH心率不齐数据库](https://www.physionet.org/content/mitdb/1.0.0/)。麻省理工学院-BIH心律失常数据库包含48个半小时的双通道动态心电图记录摘录，这些记录来自BIH心律失常实验室在1975年至1979年间研究的47个受试者。从波士顿贝丝以色列医院从混合住院患者（约60%）和门诊患者（约40%）中随机收集的4000张24小时流动心电图记录中随机选择23张；其余25张记录从同一组中选择，以包括不太常见但临床意义的心律失常，这些心律失常在小型随机样本中无法很好地呈现。



### 1.2心电数据格式

​		MIT为了节省文件长度和存储空间，使用了自定义的格式，一个心电记录由三个部分组成：	100/100.hea 100.dat

1. 头文件[.hea]，记录编号，导联数 采样率 采样点数等
2. 数据文件[.dat]，心电信号
3. 标记文件[.atr]，人工标注的心拍位置和类型

>头文件（.hea）储存方式为ASCLL码字符

以record 100 为例，头文件为：

```python
100 2 360 650000
'''
   100: 文件编号名称。
     2: 样本个数。MIT-BIH的心电数据是二导联数据，分别为矫正肢体导联II（modefied limb lead II, MLII）和矫正导联V1/V2/V5/V4。所以样本数目是2。
   360:采样率，采样率是每秒采集的信号点个数，就是360个采样点/秒。
650000:信号长度（采样点个数），一共650000，时间为650000/360秒。
'''
100.dat 212 200 11 1024 995 -22131 0 MLII

100.dat 212 200 11 1024 1011 20052 0 V5
'''
     212:数据格式。
     200:信号增益。200ADC（模拟数字转换器）units/mV。信号增益是信号的放大，采集人体心率信号将其放大，这里将每1mV的电压经过ADC转换成200值的数字信号。
      11:ADC转换的分辨率，ADC能够分辨量化的最小信号能力。
    1024:ADC的零值为1024，可以认为是基线值。
995/1011:两个信号的第一采样点的值。
后两个值分别为采样点的检验数和输入输出的块的尺寸信息。
'''
# 69 M 1085 1629 x1
 
# Aldomet, Inderal
'''
包括数据来源患者的基本情况以及用药信息。
'''
```

​		每个PhysioNet中的生理信号数据都有其详细说明和使用指南，可以自行阅读。例如[MIT-BIH数据库说明](http://www.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm)。[PhysioBank ATM](https://archive.physionet.org/cgi-bin/atm/ATM)则是提供了浏览各种生理数据库的平台，可以在这个网页上获取到生理信号数据，可视化图片等。![截屏2020-12-07 下午3.46.47](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-07 下午3.46.47.png)

​		类似的数据库还有标注文件[.qrs]，这类文件只有对心拍位置的人工标记，并没有标注是何种心拍。





### 1.3 心电数据读取

+ python中需要使用`wfdb`包进行数据的读取，[wfdb包资料](https://pypi.org/project/wfdb/)，可自行浏览。以下进行MIT-BIH心电数据记录100的信号读取。

```python
# 读取编号为data的一条心电数据
def read_ecg_data(data):
    '''
    读取心电信号文件
    sampfrom: 设置读取心电信号的起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的结束位置，sampto = 1500表示从1500出结束，默认读到文件末尾
    channel_names：设置设置读取心电信号名字，必须是列表，channel_names=['MLII']表示读取MLII导联线
    channels：设置读取第几个心电信号，必须是列表，channels=[0, 3]表示读取第0和第3个信号，注意信号数不确定
    '''
    # 读取所有导联的信号
    record = wfdb.rdrecord('../ecg_data/' + data, sampfrom=0, sampto=1500)
    # 仅仅读取“MLII”导联的信号
    # record = wfdb.rdrecord('../ecg_data/' + data, sampfrom=0, sampto=1500, channel_names=['MLII'])
    # 仅仅读取第0个信号（MLII）
    # record = wfdb.rdrecord('../ecg_data/' + data, sampfrom=0, sampto=1500, channels=[0])

    # 查看record类型
    print(type(record))
    # 查看类中的方法和属性
    print(dir(record))

    # 获得心电导联线信号，本文获得是MLII和V1信号数据
    print(record.p_signal)
    print(np.shape(record.p_signal))
    # 查看导联线信号长度，本文信号长度1500
    print(record.sig_len)
    # 查看文件名
    print(record.record_name)
    # 查看导联线条数，本文为导联线条数2
    print(record.n_sig)
    # 查看信号名称（列表），本文导联线名称['MLII', 'V1']
    print(record.sig_name)
    # 查看采样率
    print(record.fs)

    '''
    读取注解文件
    sampfrom: 设置读取心电信号的起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的结束位置，sampto=1500表示从1500出结束，默认读到文件末尾
    '''
    annotation = wfdb.rdann('../ecg_data/' + data, 'atr')
    # 查看annotation类型
    print(type(annotation))
    # 查看类中的方法和属性
    print(dir(annotation))

    # 标注每一个心拍的R波的尖锋位置的信号点，与心电信号对应
    print(annotation.sample)
    # 标注每一个心拍的类型N，L，R等等
    print(annotation.symbol)
    # 被标注的数量
    print(annotation.ann_len)
    # 被标注的文件名
    print(annotation.record_name)
    # 查看心拍的类型
    print(wfdb.show_ann_labels())

    # 画出数据
    draw_ecg(record.p_signal)
    # 返回一个numpy二维数组类型的心电信号，shape=(65000,1)
    return record.p_signal

```

+ 使用`pywt`进行滤波处理。
+ ECG信号具有微弱、低幅值、低频、随机性的特点，很容易被噪声干扰，而噪声可能来自生物体内，如呼吸、肌肉颤抖，也可能因为接触不良而引起体外干扰。是ECG信号主要的三种噪声为工频干扰、肌电干扰和基线漂移，也是在滤波过程中急需被抑制去除的噪声干扰。
+ 工频干扰：是由采集心电信号的设备周身的供电环境引起的电磁干扰，幅值低，噪声频率为50Hz左右，其波形很像一个正弦信号，该噪声常常会淹没有用的心电信号，也会影响P波和T波的检测。

![20200506221121](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20200506221121.jpg)

+ 肌电干扰：在心电图采集过程中，因为人体运动肌肉不自主颤抖造成，这种干扰无规律可言，波形形态会急速变化，频率很高，并且分布很广，范围在0-2000Hz内，能量集中在30-300Hz内，持续时间一般为50ms，肌电干扰与心电信号会重合在一起，这会导致有用的心电信号细微的变化很可能被忽视。

![20200507004744](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20200507004744.jpg)

+ 基线漂移：属于低频干扰，频率分布在0.15-0.3Hz内，由于电极位置的滑动变化或者人体的呼吸运动造成心电信号随时间缓慢变化而偏离正常基线位置产生基线漂移，幅度和频率都会时刻变化着。心电信号中的PR波段和ST波段非常容易受到影响产生失真。

![20200506221528](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20200506221528.jpg)

### 1.4 心电预处理

> 小波变换（Wavelet Transform, WT）可以进行时频变换，是对信号进行时域以及频域分析的最为理想工具。本文对含噪心电信号采用基于小波变换的去噪处理方法，分为以下3个步骤：

1. 由于噪声和信号混杂在一起，首先选择一个小波基函数，由于噪声和信号混杂在一起，所以要用小波变换对含噪心电信号进行某尺度分解得到各尺度上的小波系数。
2. 心电信号经过小波变换尺度分解后，幅值比较大的小波系数就是有用的信号，幅值比较小的小波系数就是噪声，根据心电信号和夹杂噪声的频率分布，对各尺度上的小波系数进行阈值处理，把小于阈值的小波系数置零或用阈值函数处理。
3. 分别处理完小波尺度分解后的低频系数和高频系数，再重构信号。

![20200506222110](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20200506222110.png)

4. 尺度小波分解所得各尺度系数示意图如下，9尺度小波分解可以类比之：

![20200506233227](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20200506233227.png)



> 小波系数处理的阈值函数有硬阈值和软阈值之分。

1. 硬阈值函数：若分解后的系数绝对值大于阈值，保证其值不变；当其小于给定的阈值时，令其为零。

2. 软阈值函数：若分解后的系数绝对值大于阈值，令其值减去λ；当其小于给定的阈值时，令其为零。

其中w为原始小波系数，W为处理后的小波系数，λ为给定的阈值，N为信号长度。λ的计算公式为
$$
\lambda=\frac{median|w|\sqrt{2lnN}}{0.6745}
$$

+ [傅立叶变换与小波变换](https://blog.csdn.net/tbl1234567/article/details/52662985)

+ python预处理心电数据

```python
record = wfdb.rdrecord(100, channel_names=['MLII'])
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
plt.figure(figsize=(20, 4))
plt.subplot(2,1,1)
plt.plot(data)
plt.subplot(2,1,2)
plt.plot(rdata)
plt.show()

```

![myplot](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/myplot.png)







## 2、心电数据融合

+ 由于各个实验室采集数据时采用的导联方式不同、采样频率不同、数据标签不同，因此在合并时需要对其进行统一。最终我们将数据统一成500Hz，切分成10秒，作为神经网络的输入数据。至于不同的导联方式则不加以区分，而是在网络开始处设计卷积层，自动提取不同的导联方式特征。
+ 不同数据采样频率融合的方式一般采用插值（频率不足500Hz）、下采样（频率高于500Hz）。
+ 不同数据分类的融合方式一般采用分层采样法，即保证每一种分类的数据都处在一个相对平衡的状态。

|          数据集          |                 来源                  | 原始频率/Hz | 患者数/个 | 切割样本数/个 | 导联数 |
| :----------------------: | :-----------------------------------: | :---------: | :-------: | :-----------: | ------ |
|      MIT-BIH数据集       |   美国麻省理工学院、Beth Israel医院   |     360     |    47     |     7372      | 2      |
|   PTB-XL大型心电数据集   |            德国国家计量署             |     500     |   18885   |  21837 * 12   | 12     |
| 心脏性猝死动态心电数据集 | 2004年PhysioNet“心脏病学计算机挑战赛” |     250     |    23     |               | 2      |
|       AF房颤数据集       |               PhysioNet               |     128     |    10     |      180      | 2      |

### 2.1 MIT-BIH数据集

> 数据库出处：https://physionet.org/content/mitdb/1.0.0/

#### 2.1.1 数据集简介

* 美国的MIT-BIH心电数据库是**目前在国际上应用最多的数据库**，由很多**子数据**库组成，每个子数据库包含某类特定类型的心电记录，其中应用的最多的是MIT-BIT心律不齐数据库。自1999年，在美国国家研究资源中心和国家健康研究院的支持下，他们将该数据库公开到了Internet上，整个MIT-BIT数据库的所有数据都可以免费下载和使用，国内外许多心电方面的研究都是基于该数据库的，使用该数据库作为实验数据的来源和各类识别算法的检测标准。
* MIT-BIH心律失常数据库包含48个长度为30分钟的动态心电图记录片段，这些片段是从1975~1979年间BIH心律失常实验室的47名研究对象那里获得的。其中23个记录是从一个包含4000个24小时动态ECG记录的数据集中随机选取的，这些动态ECG记录是波士顿Beth Israel医院从住院病人（约60%）和门诊病人（约40%）这样一个混合人群里收集的。其余25个记录是从同一个数据集中选取的，用来包含那些不常见的，在一个小随机样本中不能很好的表示的心律失常。
* MIT-BIH心率失常数据库原始采样频率为360Hz，分辨率为11bit，导联数为2。

#### 2.1.2 数据处理

```{python}
import wfdb
import pywt
import matplotlib.pyplot as plt
import seaborn
import numpy as np

# 要读取的数据集存放的路径
PATH = 'dataset/1.MIT-BIH/'

Index1 = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', 
          '113', '114', '115', '116', '117', '119', '121', '122', '123', '124', 
          '200', '201', '202', '203', '205', '208', '210', '212', '213', '214', 
          '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', 
          '232', '233', '234']
Index2 = ['00735','03665','04015','04043','04048','04126','04746','04908','04936',
          '05091','05121','05261','06426','06453','06995','07162','07859','07879',
          '07910','08215','08219','08378','08405','08434','08455']
Index3 = ['800','801','802','803','804','805','806','807','808','809','810','811',
          '812','820','821','822','823','824','825','826','827','828','829','840',
          '841','842','843','844','845','846','847','848','849','850','851','852',
          '853','854','855','856','857','858','859','860','861','862','863','864',
          '865','866','867','868','869','870','871','872','873','874','875','876',
          '877','878','879','880','881','882','883','884','885','886','887','888',
          '889','890','891','892','893','894']


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def loaddata(name):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    X_data=[]
    Y_data=[]
    
    # 按照需求读取不同子库
    if name == "心律失常":
        numbers = Index1
    elif name == "心房颤动":
        numbers = Index2
    elif name == "室上性心律失常":
        numbers = Index3
    else:
        print("error")
    
    
    # 读取每一条数据
    for number in numbers:
        file = wfdb.rdrecord(PATH + number, channel_names=['MLII'])
        data = file.p_signal.flatten()
        
        # 小波变换降噪
        data = denoise(data)
        
        # 读取.atr文件中的标签信息
        annotation = wfdb.rdann(PATH + number, 'atr')
        Rlocation = annotation.sample
        Rclass = annotation.symbol
        
        # 去除首尾冗杂数据
        start = 10
        end = 10
        i = start
        j = len(annotation.symbol) - end
        while i<j:
            try:
                # 通过R波位置进行定位
                lable = ecgClassSet.index(Rclass[i])
                # 从R波往前取99个点，往后取3500个点，一共构成3600个点
                x_train = data[Rlocation[i] - 99:Rlocation[i] + 3501]
                X_data.append(x_train)
                Y_data.append(lable)
                # 两条数据间间隔10个周期
                i += 10
            except ValueError:
                i += 10
    
    # 整个X和Y的数据
    del_index=[]
    for i in range(len(X_data)):
        if len(X_data[i]) != 3600:
            del_index.append(i)
    X_data = [X_data[i] for i in range(0, len(X_data), 1) if i not in del_index]
    Y_data = [Y_data[i] for i in range(0, len(Y_data), 1) if i not in del_index]
    X_data = np.array(X_data)
    Y_data = np.array(Y_data).reshape(len(Y_data),1)
    new = np.append(X_data,Y_data,axis=1)
    
    return new
    
new = loaddata("心律失常")
np.savetxt("data.csv", new, delimiter=',')
```

### 2.2 PTB-XL大型心电数据集

> 数据库出处：https://physionet.org/content/ptb-xl/1.0.1/

#### 2.2.1 数据集简介

* PTB-XL数据集是德国国家计量署提供的数字化心电数据集，目的在于算法的研究与教学。数据来自本杰明富兰克林医学大学的心脏内科。它是一个大型数据集，包含来自18885名患者的21837个临床12导联ECG，其中52％是男性，48％是女性，年龄范围从0到95岁。
* 原始采样频率为500Hz，数据处理中下采样至360Hz；导联数为12
* 该数据集主要患病类型的分布如下：

|             类型              | 患者数 |
| :---------------------------: | :----: |
|           正常NORM            |  9528  |
| 心肌梗塞Myocardial Infarction |  5486  |
|           ST/T异变            |  5250  |
|             肥大              |  2655  |

#### 2.2.2 数据处理

```python
import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f, channels=[0]) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f, channels=[0]) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def convert(variable):
    new_array=[]
    for i in variable:
        print(i)
        for j in i:
            print(j)
            new_array.append(j)
    return new_array


path = ''
sampling_rate = 360

Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id', encoding='GB2312')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# 读取数据
X = load_raw_data(Y, sampling_rate, path)

agg_df = pd.read_csv('scp_statements.csv', index_col=0, encoding='GB2312')
agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# 读取标签类别
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
# 标签
label = Y.diagnostic_superclass.values
label = convert(label)
label = np.array(label)
np.savetxt("label.txt", label, fmt="%s")

np.savez("data.txt", X)   # 保存三维数组
print(X.shape)


# 降采样
def downsample(data, prehz, newhz):
    time = int(len(data)/prehz)  # 时间长度s
    st = float(prehz)/float(newhz)
    new_series = []
    for i in range(0, time):
        pre_series = data[i*prehz:(i+1)*prehz-1]  # 每1秒的数据
        temp = []         # 存放新1s的数据
        temp.append(pre_series[0])
        for j in range(1, newhz - 1):
            index = round(j * st)
            temp.append(data[index])
        temp.append(pre_series[-1])
        new_series.append(temp)
    return new_series


data = np.load('data.txt.npz')['arr_0']
data = np.reshape(data, (21837, -1))


# 转换成360hz并归一化至0-1
def hz_convert(array):
    new_array = []
    for i in range(0, len(array)):
        each = array[i]
        each = downsample(each, 500, 360)
        each = np.reshape(each, (-1, 1))
        each = MinMaxScaler(feature_range=(0, 1)).fit_transform(each)
        new_array.append(each.flatten())
    return new_array


new = np.array(hz_convert(data))
print(new.shape)
plt.plot(new[0])
plt.plot(new[1])
plt.show()
np.savetxt("channel_1.txt", new)
```

### 2.3 AF房颤数据集

> 数据库出处：https://physionet.org/content/aftdb/1.0.0/

#### 2.3.1 数据集简介

* 该数据集从公开数据集中提取，每条记录都是一分钟的房颤。数据分为**NST三组**，分别来自同一次房颤发作的不同时期：
  * N组：非终止AF（定义为在该段记录后**至少一个小时内未观察到终止**）
  * S组：在记录结束**后一分钟终止**
  * T组：记录结束后**立即终止**（这些记录来自与S组相同的长期ECG记录，并且是对应S组记录的延续）

* 数据集存有10名房颤患者的数据，原始采样频率为128Hz，数据处理中上采样至360Hz；导联数为2

#### 2.3.2 数据处理

```python
from scipy import interpolate
import wfdb
import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


PATH = 'learning-set/'
classSet = ['n', 's', 't']
numberSet = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def interpolation(data, prehz, newhz):
    y = np.array(data)
    x = np.linspace(0, len(data), len(data))
    time = len(data)/prehz
    addn = int((newhz - prehz) * time)
    x_new = np.linspace(0, len(data), len(data)+addn)
    f = interpolate.interp1d(x, y, kind='cubic')
    y_new = f(x_new)
    return y_new

X_data = []
Y_data = []

# 切割
def cut_series(data, time, fs):
    cut_length = time*fs
    number = int(len(data)/cut_length)
    # print(number)
    for i in range(0, number):
        cut = data[i*cut_length:(i+1)*cut_length].reshape(-1, 1)
        cut = MinMaxScaler(feature_range=(0, 1)).fit_transform(cut)
        print(cut)
        plt.plot(cut)
        plt.show()
        X_data.append(cut.flatten())


def read_single_data(type, number):
    file_path = PATH + type + number
    record = wfdb.rdrecord(file_path, channel_names=['ECG'])  # 读取
    rdata = record.p_signal.flatten()
    data = denoise(rdata)   # 去噪
    new_data = interpolation(data, 128, 360)   # 上采样
    cut_series(new_data, 10, 360)     # 切割


def read_all_data():
    for type in classSet:
        for number in numberSet:
            read_single_data(type, number)


read_all_data()
X_data = np.array(X_data)
print(X_data.shape)
np.savetxt("data.txt", X_data)
```

> 类似的数据处理有很多就不在这里一一展示。





## 3、Tensorflow模型构建（ECGNet）

> 此模型训练的是单导联模型（将数据库中的所有导联方式作为样本）

### 3.1模型总览

**模型总览**

![截屏2020-12-07 下午9.15.09](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-07 下午9.15.09.png)

​		这次试验中，我们使用的是适用多导联心电图数据的多尺度ResNet模型：ECGNet。总体思想如下：

1、利用不同导联之间的相似性

​		不同心电图数据有不同的导联方式，我们如何处理不同导联数据之间的差异？那么我们首先要寻找不同导联之间的相似性，将不同导联数据之间的相似特征放大提取，这样就可以对利用不同的导联方式获取的心电图数据进行处理，而不必考虑不同导联数据之间的差异性，从而使得这个模型泛化性提高。



![图片 1](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/图片 1.png)



​		具体做法是：在搭建网络结构过程的初期，我们将一维的心电图数据当作二维的数据进行处理，具体表现在将一维数据转化为二维数据。其次，为了获得不同导联数据中的相似性，我们在不同导联方式中获取的数据初期采用相同的卷积核。以上的处理作用是：A、提取出不同导联方式数据之间的共性特征，从而能够对不同导联方式获取到的数据进行处理，提高模型的泛化能力；B、对不同导联数据使用相同的卷积核，减少了参数量，减少模型训练成本以及模型预测速度。



![导联相似性](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/导联相似性.png )



​		使用了相同的卷积核操作不同的导联数据后，模型将使用多尺度卷积的方式，能提取出不同导联之间的差异性特征，后期将不同导联提取出来的特征合并，重新变为一维数据。

2、卷积核长度先长后短

![卷积核变化](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/卷积核变化.png)

​		处理ECG信号时，前期我们先采用较长的卷积核，因为对于采样频率高的数据来说，采用较长的卷积核取得效果较好。对于不同导联方式，我们想要通过卷积获得他们的相似性，需要从大范围上获取，因此使用较长的卷积核效果较好。

​		随着特征尺寸的减小，我们不断减小卷积核长度。当到了多尺度卷积中，为了提取出不同导联之间的差异性特征，依然适当减小卷积核的长度以便于获取不同导联中细节的特征。不仅如此，使用小的卷积核长度同时可以减小运算量。



3、改变传统模型结构：Pre-Activation

​		Post和Pre的概念是针对weight（conv）层来说的，我们在实验过程中发现BN-ReLU-Conv结构是要优于传统结构Conv-BN-ReLU的。

![preactivation](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/preactivation.png)

[1]He, Kaiming, et al. "Identity mappings in deep residual networks." *European conference on computer vision*. Springer, Cham, 2016.



4、SE-ResNet结构构建

​		作为模型中的子结构，可以很方便地嵌入到其他分类或检测模型中。这个子结构的作用是提高模型对重要特征的“重视程度”，也就是将所需要的特征更加突出。在这个模型中他是针对通道的注意力模型，学习了通道的重要程度，分为Squeeze和Excitation两个部分。

![squeeze](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/squeeze.png)



5、多尺度网络结构

​		ECGNet模型中，我们使用并行多分支网络。并行结构能够在同一层级获取不同感受野的特征，经过融合后传递到下一层，可以更加灵活的平衡计算量和模型能力。在这个模型中表现为提取出不同导联之间的差异特征处，后期将不同导联提取出来的特征合并，重新变为一维数据。



6、其他细节

![截屏2020-12-07 下午9.47.51](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-07 下午9.47.51.png)





模型亮点：

（1）利用导联的相似性；

（2）初期对不同导联应用相同的卷积核，能在减小参数量的同时很好的提升模型的鲁棒性；

（3）采用多尺度网络结构能捕捉不同尺度的特征，较好的提升模型的效果；

（4）BN-ReLU-Conv优于Conv-BN-ReLU；

（5）SE-ResNet结构能提升模型的效果。





### 3.2使用Tensorflow构建ECGNet

```python
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
```





## 4、模型训练

### 4.1服务器配置

>由于模型规模庞大，无法使用本地电脑CPU训练模型，我们使用服务器GPU训练模型。该服务器包含显卡：
>
>4 * Tesla V100 PCI-E 32G

#### 4.1.1 Anaconda下载与安装

官网地址:[https://www.anaconda.com/products/individual#download-section](https://www.anaconda.com/products/individual#download-section)

清华镜像:[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

下载linux系统的Python 3.7 version

下载结束后将其ftp传至服务器，并使用以下命令进行安装

```
bash Anaconda3-2019.03-Linux-x86_64.sh
```

![截屏2020-12-08 下午8.12.37](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午8.12.37.png)

+ 输入`yes`

![截屏2020-12-08 下午8.13.32](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午8.13.32.png)

+ 选择安装位置，可以默认`enter`，也可以自行选择位置。

![截屏2020-12-08 下午8.16.37](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午8.16.37.png)

+ 继续输入`yes`。

+ 输入`conda --version`查看安装是否成功，若显示`未找到命令`，命令行输入`source ～/.bashrc`。

#### 4.1.2 构建Tensorflow-gpu环境

+ 建立Tensorflow-gpu环境并进入虚拟环境。

```
conda create -n tensorflow-gpu(env-name)
conda activate tensorflow-gpu
```

+ 命令行前头有`(tensorflow-gpu)`则表示在虚拟环境中。

+ 退出虚拟环境，`conda deactivate`。



#### 4.1.3CUDA、cudnn

> 由于使用的服务器中已配置好显卡驱动，故这里不进行展示

+ 安装好显卡驱动后，命令行输入`nvidia-smi`，可以看见显卡状态

![截屏2020-12-08 下午8.01.18](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午8.01.18.png)



### 4.2服务器GPU训练模型

> 使用CUDA_VISIBLE_DEVICES=0,1,2,3指定使用GPU

#### 4.2.1 训练模型

```python
conda activate tensorflow-gpu # 进入tensorflow-gpu环境
CUDA_VISIBLE_DIVICES=0,1,2,3 python ECGnet.py
```

+ 模型部分

![截屏2020-12-08 下午8.04.01](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午8.04.01.png)

+ 模型训练中

![截屏2020-12-08 下午8.07.55](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午8.07.55.png)



#### 4.2.2 模型效果

![图片4](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/图片4.png)

![图片5](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/图片5.png)



+ 可以看到ECGNet模型训练的效果是比较好的，测试集也能达到100%准确率。





## 5、模型移植树莓派

### 5.1模型转换

+ 因为Tensorflow模型较为庞大，以树莓派的算力不足以支撑模型的运行，故将模型转换成Tensorflow-Lite模型进行模型移植

模型转换主要代码如下：

```python
import tensorflow as tf
converter=tf.lite.TFLiteConverter.from_keras_model(model)
# model 为ECGNet模型
tflite_model = converter.convert()
tflite_model_file ="./model/ecgmodel.tflite"
with open(tflite_model_file,"wb") as f:
		f.write(tflite_model)
# 保存模型
interpreter=tf.lite.Interpreter(model_path="./model/ecgmodel.tflite")
# 加载模型
interpreter.allocate_tensors()
# 为编译器分配空间
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
data = data.reshape((1,5000,1,1))
# 将预测样本大小转换成模型输入大小
data = data.astype(np.float32)

# tflite模型进行预测
interpreter.set_tensor(input_index, data)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)
print(predictions)
```

> 至此我们就得到了ecgmodel.tflite模型，后续将在树莓派上调用这个模型进行预测。





### 5.2树莓派配置

#### 5.2.1 树莓派写入桌面版系统

#### 5.2.2 树莓派安装python3

> 因为树莓派默认系统内的python版本是python2，但由于我们的模型的运行需要在python3的环境下运行，所以首先安装python3.7

##### 5.2.2.1 安装依赖包

```c
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev   
sudo apt-get install -y libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm 
sudo apt-get install -y libncurses5-dev  libncursesw5-dev xz-utils tk-dev
```

##### 5.2.2.2 下载python3.7

```c
sudo wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz
```

+ 也可以从[python网页](https://www.python.org/ftp/python/)，中寻找需要的版本

##### 5.2.2.3 解压python3.7

```c
sudo tar -zxvf Python-3.7.3.tgz
```

##### 5.2.2.4 进入解压包并安装python

```c
cd Python-3.7.3
sudo ./configure --prefix=/usr/local/python3
sudo make
```

##### 5.2.2.5 安装完成，创建软连接

```c
ln -s /usr/local/python3/bin/python3 /usr/local/bin/python3
ln -s /usr/local/python3/bin/pip3 /usr/local/bin/pip3
```

+ 有时会出现软连接已存在问题，需要删除原有软连接

```c
rm -rf /usr/local/bin/python3
rm -rf /usr/local/bin/pip3
```

##### 5.2.2.6 打印版本测试

```c
python3 -V
pip3 -V
```

![20191230113256692](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20191230113256692.png)

> 安装完成！

#### 5.2.3 安装Tensorflow2.0

##### 5.2.3.1 检查配置相关环境

```c
python3 --version
pip3 --version
virtualenv --version
```

+ 若某一条运行时提示错误，则需要以下操作

```python
sudo apt update # 更新
sudo apt install python3-dev python3-pip  # 如已经安装了Python3和pip3则跳过此命令
sudo apt install libatlas-base-dev  # 此命令必选
sudo pip3 install -U virtualenv  # 如果已经安装了虚拟环境，跳过此命令
```

##### 5.2.3.2 创建虚拟环境并激活虚拟环境

```c
virtualenv --system-site-packages ./venv
source ./venv/bin/activate
```

##### 5.2.3.3 pip转换国内源，更新pip并且安装依赖包

```python
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip #更新pip
# 安装依赖包
pip install keras_applications==1.0.8 --no-deps
pip install keras_preprocessing==1.1.0 --no-deps
pip install h5py==2.9.0
pip install -U six wheel mock
```

##### 5.2.3.4 下载Tensorflow

+ 在一个github上下载[Tensorflow2.0](https://github.com/lhelontra/tensorflow-on-arm/releases)，桌面版系统树莓派可以直接联网下载，命令行版系统树莓派需要外界下载后将下载包传递至树莓派内。

+ 在这里我们下载的是Tensorflow2.0.0，（需要对照自己树莓派版本下载）。将Tensorflow2.0.0下载至树莓派后，安装Tensorflow。

```c
pip install tensorflow-2.0.0-cp37-none-linux_armv6l.whl
```

+ 由于树莓派0内存较小，在运行安装过程中经常出现**Memory Error**的错误，可用以下命令解决：

```python
pip3 install --no-cache-dir tenstoflow-2.0.0-cp37-none-linux_armv6l.whl #--no-cache-dir 参数表示禁用缓存
```

+ 安装结束后可以打开python，`import tensorflow as tf`测试，未报错则安装完成。

#### 5.2.4 其他包安装

+ （还是在虚拟环境下）,命令行前头有`(venv)`则表示在虚拟环境中

```python
pip install matplotlib # 画图
pip install Adafruit_ADS1x15 # 数模转换器
pip install sklearn # 使用sklearn里面的数据处理模块
```

### 5.3其他硬件配置

#### 5.3.1 ADC模块ADS1115数模转换模块

> 由于使用的是AD8232心电模块采集外部信号，树莓派没有专门处理生理信号的接口，所以需要ADC1115数模转换模块将生理信号转换成数字信号。

**接线总览**

+ 3.3v（树莓派）----- VDD（ADS1115） -----3.3v（AD8232）
+ GND（树莓派） ----- GND（ADS1115） ----- GND（AD8232）
+ SDA（树莓派）----- SDA（ADS1115）
+ SCL（树莓派） ----- SCL（ADS1115）
+ A3（ADS1115） ----- OUTPUT（AD8232）

![截屏2020-12-08 下午7.30.13](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午7.30.13.png)

+ ADS1115和树莓派之间是通过I2C总线通信，所以需要先打开树莓派I2C总线的权限。

```c
sudo raspi-config
```

![20191222162151379](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20191222162151379.png)

![20191222162151389](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20191222162151389.png)

![20191222162151394](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/20191222162151394.png)

![2019122216214693](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/2019122216214693.png)

+ 检查是否配置好

```
sudo i2cdetect -y l
```

+ 若如上图连接数模转换器，则会显示48地址。
+ 利用python使ADS1115和树莓派通信：`simplest.py`。

```python
import time
import Adafruit_ADS1x15 as ada

adc = ada.ADS1115()

while True:
    value=[0]*4
    for i in range(4):
        value[i] = adc.read_adc(i, 1)
    print('|{0:>6}|{1:>6}|{2:>6}|{3:>6}|'.format(*value))
    time.sleep(0.5)
```



#### 5.3.2 AD8232

​		AD8232是一款用于ECG及其他生物电测量应用的集成信号调理模块。该器件设计用于在具有运动或远程电极放置产生的噪声的情况下提取、放大及过滤微弱的生物电信号。该设计使得超低功耗模数转换器(ADC)或嵌入式微控制器能够轻松地采集输出信号。

![截屏2020-12-08 下午7.42.52](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/截屏2020-12-08 下午7.42.52.png)

+ 贴好电机片后就可以进行实验了，注：**电极片为一次性，撕下后无法再次使用**。

```python
python continous.py
```

![图片2](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/图片2.png)

+ 将心电模块AD8232各个导联线连接至上图所示的身体部位，给模块通电后便开始获取人体心电信号，该模块内部的高信号增益，放大电路，以及可调低通滤波器可以对人体的心电信号进行初步的采集与处理。（**上面的信号是原始信号，下方的信号是去噪后的信号**）

## 6、测试

+ 使用树莓派作为心电模块信号处理的主要载体，当心电模块采集完毕心电数据后，会将心电生理信号转化为数字信号传入树莓派，树莓派使用预先部署好的模型将数据用图表形式在显示屏上展示，并输出最终测试者的心脏状况。
+ 使用`ecgtest2.py`进行实时测试

```python
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

```

![图片3](/Users/ethan-q/PycharmProjects/DeepLearning/深度学习/图片3.png)

## 7、代码说明

+ DeepLearning
  + checkpoint：模型训练记录点
  + data：数据
  + figure：模型训练图片
  + log：模型训练记录
  + model：模型
    + ECGNet.pb：12导联模型
    + ECGNet.pb：单导联模型
    + model.tflite：tflite单导联模型（树莓派用）
  + 树莓派代码（用于树莓派上）
    + continous.py：连续获取心电信号代码，并绘制图
    + ecgdata.txt：心电数据（自己获取的）
    + ecgtest1.py：tflite模型测试代码
    + ecgtest2.py：实时心电数据监代码
    + model.tflite：tflite单导联模型（树莓派用）
    + simplest.py：ADS1115与树莓派通信实例
  + ECGNet1.py：12导联模型代码
  + ECGNet2.py：单导联模型代码
  + MIT-BIH数据读取实例.py
  + tflite模型运行实例.py
  + 单导联ECGNet1.py



> data文件夹中的使用的部分数据在百度云（因为github上传限制）：
>
> 链接：https://pan.baidu.com/s/1AR_IN4_4STTZCSSAo0MHkw 
> 提取码：61y8 
> 复制这段内容后打开百度网盘手机App，操作更方便哦

