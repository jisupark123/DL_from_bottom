# 수학 계산

import numpy as np


# 미분
# 근사치를 구하는 것이기 때문에 수치 미분이라고 한다.
# def derivative(f, x):
#     h = 1e-4  # 0.0001
#     return (f(x + h) - f(x)) / h


# 미분
# 근사치를 구하는 것이기 때문에 수치 미분이라고 한다.
# 수치 미분은 오차가 발생할 확률이 높기 때문에 x+h ~ x-h의 기울기를 구한다. (중심 차분)
def derivative(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


# 편미분 함수의 기울기 계산
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


# def numerical_gradient(f: callable, x: np.ndarray):
#     if x.ndim == 1:
#         return _numerical_gradient_no_batch(f, x)
#     else:
#         grad = np.zeros_like(x)

#         for idx, x in enumerate(x):
#             grad[idx] = _numerical_gradient_no_batch(f, x)

#         return grad


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad
