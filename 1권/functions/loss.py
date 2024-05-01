# 손실 함수

import numpy as np


# MSE 평균제곱오차
def mean_squeard_error(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum((y - t) ** 2)


# # 교차 엔트로피 오차
# def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))


# 교차 엔트로피 오차
# 배치 데이터를 지원하도록 구현
def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    # [1,2,3] -> [[1,2,3]]
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
