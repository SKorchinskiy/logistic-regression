import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List

SigOut = Union[List[float], float]

def plot_data(X_train: list, y_train,):
    plt.scatter(
        [x[0] for x in X_train], 
        [x[1] for x in X_train], 
        c=y_train, 
        cmap='coolwarm', 
        marker='x', 
        linewidths=5, 
        s=100, 
    )
    plt.xlim(0, 4)
    plt.ylim(0, 4)

    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    
    plt.title('Dataset distribution')

    plt.legend()

    plt.show()

def sigmoid(z: SigOut) -> SigOut:
    g = 1 / (1 + np.exp(-z))
    
    return g

def compute_cost_logistic(X, y, w, b) -> float:
    nX = np.array(X)
    nw = np.array(w)

    m = nX.shape[0]
    cost = 0.0

    for i in range(m):
        z_i = np.dot(nX[i], nw) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1-f_wb_i)

    cost /= m
    
    return cost

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

plot_data(X_train, y_train)


w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))