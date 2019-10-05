import numpy as np
import matplotlib.pyplot as plt

w1 = np.array([[0.5], [0.5]])
b1 = np.array([[0], [1]])
w2 = np.ones((2, 2))
b2 = np.zeros((2, 1))
w3 = np.ones((1, 2))
b3 = 1


def h(x:float) -> float:
    v1 = w1 * x + b1
    v2 = w2 @ np.maximum(0, v1) + b2
    v3 = w3 @ np.maximum(0, v2) + b3
    return v3[0][0]


if __name__ == '__main__':
    xs, vs = [], []

    for p in [1, -1, -0.5]:
        xs, vs = [], []
        for e in range(10):
            e /= 20
            xs.append(p + e)
            vs.append(h(p + e))
            xs.append(p - e)
            vs.append(h(p - e))
        xs = np.array(xs)
        vs = np.array(vs)
        b = (len(xs) * (xs * vs).sum() - xs.sum() * vs.sum()) / (len(xs) * (xs**2).sum() - xs.sum()**2)
        a = vs.sum() / len(xs) - b * xs.sum() / len(xs)

        print("at {}, h({}) = {}, h(x) = {} * x + {}".format(p, p, h(p), b, a))

    # for x in range(-40, 20, 1):
    #     x = x / 10.0
    #     v = h(x)
    #     xs.append(x)
    #     vs.append(v)
    # plt.plot(xs, vs)
    # plt.xlabel("x")
    # plt.ylabel("h(x)")
    # plt.show()







