# 숫자 판별
import sys, os

sys.path.append(os.pardir)
import pickle
import numpy as np
from dataset.mnist import load_mnist
from vitalize_func import *


def get_data():
    # (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test


def init_network():
    # 현재 실행 중인 파일의 절대경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 파일의 경로
    file_path = os.path.join(script_dir, "sample_weight.pkl")
    with open(file_path, "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
batch_size = 100  # 배치 크기

accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]
    y_batch = predict(network, x_batch)

    p = np.argmax(y_batch, axis=1)  # 100x10 행렬의 각 행마다 확률이 가장 높은 원소의 인덱스들을 배열로 반환한다
    accuracy_cnt += np.sum(p == t[i : i + batch_size])

print(f"정확도: {accuracy_cnt/len(x)}")
