from ml import loss
import numpy as np

def train(x1, x2, y_train, lr=0.05, n_iter=100):
    dt = 'float64'
    x1 = x1.astype(dt)
    x2 = x2.astype(dt)
    y_train = y_train.astype(dt)
    a1, a2, b = np.array([0.12], dtype=dt), np.array([0.22], dtype=dt), np.array([0.62], dtype=dt) # parameters
    mse = [] # lp
    N = len(x1) 
    for i in range(n_iter):
        f = y_train - (a1*x1 + a2*x2 + b)
        a1 -= lr * (-2 * x1.dot(f).sum() / N) 
        a2 -= lr * (-2 * x2.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        mse.append(loss.MSE(a1*x1 + a2*x2 + b, y_train))

    print("lr\t: ", lr)
    print("iter\t: ", n_iter)

    return a1, a2, b, mse