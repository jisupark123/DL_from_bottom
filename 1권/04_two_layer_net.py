import numpy as np
from functions.math_tools import *
from functions.gradient_descent import *
from functions.loss import *
from functions.vitalize import *


class TwoLayerNet:
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01
    ):
        # 가중치 초기화
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    # 예측값 구하기
    def predict(self, x: np.ndarray):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 손실함수의 값 구하기
    def loss(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # 정확도 구하기
    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 기울기 구하기
    def gradient(self, x: np.ndarray, t: np.ndarray):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_w, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_w, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_w, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_w, self.params["b2"])

        return grads


if __name__ == "__main__":
    from dataset.mnist import load_mnist

    # (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )
    train_loss_list = []

    # 하이퍼파라미터
    iters_num = 10000  # 반복 횟수
    train_size = x_train.shape[0]  # 훈련 데이터 사이즈 (10000)
    batch_size = 100  # 미니배치 크기
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for _ in range(iters_num):
        # 미니배치 획득 - 훈련 데이터에서 batch_size만큼 추출
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        grad = network.gradient(x_batch, t_batch)

        # 매개변수 갱신
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        # 학습 경과 기록
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        print(train_loss_list)
