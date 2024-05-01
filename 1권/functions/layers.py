import numpy as np
from functions.vitalize import softmax
from functions.loss import cross_entropy_error


class Relu:
    def __init__(self):
        self.mask = None  # numpy 배열, 원소 값이 0 이하인 인덱스 -> True

    # if x > 0 -> x
    # else -> 0
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0  # 0 이하는 모두 0으로
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    # 1 / 1 + exp(-x)
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    # dL/dy
    # -> dL/dy * y^2 * exp(-x) = dL/dy * y(1-y)
    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    # xw + b
    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)

        return dx


# Softmax & CrossEntrophy
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax 출력
        self.t = None  # 정답 레이블(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size  # data 1개당 오차를 전파
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
