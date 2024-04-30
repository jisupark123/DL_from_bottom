# 활성화 함수
import numpy as np


# 계단 함수
def step(x):
    return np.array(x > 0, dtype=int)


# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU 함수
def ReLU(x):
    return np.maximum(0, x)


# Identity 함수
def identity(x):
    return x


# Softmax 함수
def softmax(x: np.ndarray) -> np.ndarray:
    # if x.ndim == 2:
    #     x = x.T
    #     x = x - np.max(x, axis=0)
    #     y = np.exp(x) / np.sum(np.exp(x), axis=0)
    #     return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == "__main__":
    # x = np.arange(-5.0, 5.0, 0.1)
    # y = ReLU(x)
    # plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)
    # plt.show()
    a = np.array([[1, 3]])
    print(a.ndim)
    print(softmax(a))
