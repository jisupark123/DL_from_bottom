# 신경망 구현 실습

import numpy as np
from vitalize_func import *

# X = 초기값 W = 가중치 B = 편향 Z = 결과값 Y = 최종값


# 가중치, 편향에 기본값 할당
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    # 0층 -> 1층
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # 1층 -> 2층
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # 2층 -> 3층
    a3 = np.dot(z2, W3) + b3
    y = identity(a3)

    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
