import numpy as np
from ml import Lasso, dataset
import matplotlib.pyplot as plt  
plt.rc('font', family='Malgun Gothic')

#load datasets
TRAIN = 1800
TEST = 2350 #max
dataset.MIN = 0
dataset.MAX = 2350

date = dataset.DateTerm(1).values
d1 = dataset.Scale(dataset.KOSPI200FIDX())
d2 = dataset.Scale(dataset.NASDAQ100FIDX())
d3 = dataset.Scale(dataset.SAMSUNGELECSTKP())
#train, test split data-set
train_d1 = d1[0:TRAIN]; train_d2 = d2[0:TRAIN]; train_d3 = d3[0:TRAIN]
test_d1 = d1[TRAIN:TEST]; test_d2 = d2[TRAIN:TEST]; test_d3 = d3[TRAIN:TEST]

#from ml import Lasso, dataset
train_x = np.array([train_d1, train_d2])
test_x = np.array([test_d1, test_d2])

alphas = [ #나스닥 우세
    [0.1, 0.05],
    [0.01, 0.005],
    [0.001, 0]
]

fig, axes = plt.subplots(3,2)
for r, c in np.ndindex((3,2)):
    a, b, mse = Lasso.train(train_x, train_d3, alphas[r][c], lr=0.05, n_iter=1000)
    train_pred = Lasso.predict(a, b, train_x)
    test_pred = Lasso.predict(a, b, test_x)
    axes[r, c].plot(date, d1, date, d2, date, d3)
    axes[r, c].plot(date[0:TRAIN], train_pred, linestyle=':')
    axes[r, c].plot(date[TRAIN:TEST], test_pred, linestyle='--')
    axes[r, c].legend(["코스피F", "나스닥F", "삼성전자STK", "trained", "tested"])
    axes[r, c].set_ylabel("정규화")
    axes[r, c].set_xlabel("")
    axes[r, c].set_title("mse:{0}, alpha:{1}".format(str(mse[-1])[:5], alphas[r][c]), fontsize=14)

plt.setp(axes, xticks=[], xticklabels=[])
plt.xticks([])
plt.show()