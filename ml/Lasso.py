from ml import loss
import numpy as np

def train(x_train, y_train, alpha, lr=0.05, n_iter=100):
    dt = 'float64'
    x_train = x_train.astype(dt)
    y_train = y_train.astype(dt)
    a, b= np.random.random_sample(x_train.shape[0]), np.array([0.51], dtype=dt)
    a = a.reshape((a.size, 1))
    mse = []
    N = x_train.shape[1]
    for i in range(n_iter):
        y = ((x_train*a).sum(axis=0) + b)
        f = y_train - y
        for i in range(len(a)):
            a[i] -= lr * ((-2 * x_train.dot(f).sum() / N) + alpha * (np.abs(a).sum()))
        b -= lr * (-2 * f.sum() / N)
        mse.append(loss.MSE(y, y_train))
    print("lr\t: ", lr)
    print("iter\t: ", n_iter)
    return a, b, mse
def predict(weights, b, x):
    return (weights*x).sum(axis=0) + b