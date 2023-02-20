import numpy as np
from ml import MLR, dataset
import matplotlib.pyplot as plt  
plt.rc('font', family='Malgun Gothic')

#load datasets
TRAIN = 1800
TEST = 2350 #max
dataset.MIN = 0
dataset.MAX = 2350

date = dataset.DateTerm(2).values
d2 = dataset.Scale(dataset.NASDAQ100FIDX())
d3 = dataset.Scale(dataset.SAMSUNGELECSTKP())
#train, test split data-set
train_d2 = d2[0:TRAIN]; train_d3 = d3[0:TRAIN]
test_d2 = d2[TRAIN:TEST]; test_d3 = d3[TRAIN:TEST]


a1, a2, b, mse = MLR.train(np.zeros(shape=train_d2.shape), train_d2, train_d3)

train_pred = a2*train_d2 + b
test_pred = a2*test_d2 + b
plt.plot(date, d2, date, d3)
plt.plot(date[0:TRAIN], train_pred, linestyle=':')
plt.plot(date[TRAIN:TEST], test_pred, linestyle='--')
plt.plot(date, a2*np.linspace(0, 1, dataset.MAX)+b)

plt.legend(["NASDAQ100FIDX", "SAMSUNGELECSTKP", "학습 주가 예측", "비학습 주가 예측", "주가 선형 그래프"])
plt.ylabel("정규화된 수치")
plt.xlabel("{0}--> 날짜 진행 -->{1}".format(date[0], date[-1]))
plt.xticks([], color='w')

plt.show()