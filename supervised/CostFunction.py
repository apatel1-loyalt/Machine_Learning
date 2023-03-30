import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

def cost(x_train, y_train,w, b):
    m = x_train.shape[0]
    ToalCost = 0.0
    for i in range(m):
        fw_b = w * x_train[i] + b
        ToalCost += (fw_b - y_train[i]) ** 2
    return (1 / (2 * m)) * ToalCost

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])
w = -100
b = -100
cost = 0.0

list_cost = [[]]
for i in range(100):
    for j in range(100):
        list_cost[i].append(float(cost(x_train, y_train,w, b)))
    b += 5
print(list.sort(list_cost))