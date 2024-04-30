import numpy as np
from functions.vitalize import softmax
from functions.loss import cross_entropy_error
from functions.math_tools import numerical_gradient


class SimpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)  # 2x3인 가중치 매개변수

    # 예측값 구하기
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W)

    # 손실함수의 값 구하기
    # x - 입력 데이터, t - 정답 레이블
    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])  # 입력 데이터
t = np.array([0, 0, 1])  # 정답 레이블

net = SimpleNet()

f = lambda w: net.loss(x, t)

print(numerical_gradient(f, net.W))
