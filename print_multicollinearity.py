import numpy as np
from ml import Lasso, dataset
for r in [2350, 1550, 1050]:
    dataset.MIN = 0; dataset.MAX = r
    date = dataset.DateTerm(1).values
    d1 = dataset.Scale(dataset.KOSPI200FIDX())
    d2 = dataset.Scale(dataset.NASDAQ100FIDX())
    d3 = dataset.Scale(dataset.SAMSUNGELECSTKP())
    a, b, mse = Lasso.train(np.array([d1, d2]), d3, 0, lr=0.05, n_iter=1000)