# 경사 하강법


from functions.math_tools import *

"""
f - 최적화하려는 함수
init_x - 초깃값
lr - learning rate(학습률)
step_num - 반복 횟수
"""


# 최초의 기울기 init_x가 최솟값인 기울기 x가 되어 반환된다.
def gradient_discent(f: callable, init_x: float, lr=0.01, step_num=100):
    x = init_x
    for _ in range(step_num):
        grad = gradient(f, x)  # 기울기
        x -= lr * grad
    return x


def f(x: np.ndarray):
    return x[0] ** 2 + x[1] ** 2


if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])
    res = gradient_discent(f, init_x, lr=0.1, step_num=100)
    print(res)
