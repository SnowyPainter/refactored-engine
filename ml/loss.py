import numpy as np
#y = f(x_train)
def MSE(y, y_train):
    return np.square(np.subtract(y, y_train)).mean()
def MAE(y, y_train):
    return np.absolute(np.subtract(y, y_train)).mean()