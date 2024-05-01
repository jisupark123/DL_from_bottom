# 숫자 판별
import sys, os

sys.path.append(os.pardir)
import pickle
import numpy as np
from dataset.mnist import load_mnist
from functions.vitalize import *


# 데이터셋 불러오기
def get_data():
    # (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test


# 이미 학습된 매개변수 불러오기
def init_network():
    # 현재 실행 중인 파일의 절대경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 파일의 경로
    file_path = os.path.join(script_dir, "sample_weight.pkl")
    with open(file_path, "rb") as f:
        network = pickle.load(f)

    return network


# 추론 과정
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

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스
    if p == t[i]:
        accuracy_cnt += 1

print(f"정확도: {accuracy_cnt/len(x)}")
