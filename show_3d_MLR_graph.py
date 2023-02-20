import numpy as np
from ml import MLR, dataset
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt  
plt.rc('font', family='Malgun Gothic')

fig = plt.figure()
ax = plt.axes(projection='3d')

dataset.MIN = 0
dataset.MAX = 2350

date = dataset.DateTerm(1).values
d1 = dataset.Scale(dataset.KOSPI200FIDX())
d2 = dataset.Scale(dataset.NASDAQ100FIDX())
d3 = dataset.Scale(dataset.SAMSUNGELECSTKP())

a1, a2, b, mse = MLR.train(d1, d2, d3)
c = np.linspace(0, 1, dataset.MAX)
linear_line = a1*c + a2*c + b

ax.scatter3D(d1, d2, d3)

#마땅히 쓸데 없음. 플롯 폐기.
#ax.plot(d1, d2, linear_line)

ax.set_xlabel('코스피 선물 지수')
ax.set_ylabel('나스닥 선물 지수')
ax.set_zlabel('삼성전자 주가')

plt.show()