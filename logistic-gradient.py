import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import copy, math

def plot_data(X_train: list, y_train: list, w: list, b: float, boundary_points: list):
    plt.scatter(
        [x[0] for x in X_train], 
        [x[1] for x in X_train], 
        c=y_train, 
        cmap='coolwarm', 
        marker='o', 
        linewidths=5, 
        s=100, 
    )
    plt.plot(*boundary_points, c="red", lw=1)

    m = np.array(X_train).shape[0]

    plt.plot(range(m), [sigmoid(np.dot(X_train[i,:], w) + b) for i in range(m)], c='orange')

    plt.xlim(min(X_train[:, 0]) - 0.1, max(X_train[:, 0]) + 0.1)
    plt.ylim(-0.1, 1.1)

    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    
    plt.title('Dataset distribution')

    plt.legend()

    plt.show()
    
SigOut = Union[List[float], float]

def sigmoid(z: SigOut) -> SigOut:
    g = 1 / (1 + np.exp(-z))
    
    return g


def compute_gradient_logistic(X, y, w, b):
    m, n = np.array(X).shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

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

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i<100000:
            J_history.append(compute_cost_logistic(X, y, w, b) )
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    
    return w, b, J_history

X_train = np.array([[0., 0.], [1., 0.,], [2., 0.,], [3., 1.], [4., 1.], [5., 1.]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]

print([[0,x0],[x1,0]])

plot_data(X_train, y_train, w_out, b_out, [[0,x0],[x1,0]])

plt.show()