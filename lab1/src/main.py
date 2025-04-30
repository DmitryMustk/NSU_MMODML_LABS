import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import uniform

EPS_0 = 1
POINTS = 64
FULL_POINTS = 128
POLY_DEGREE = 5


def funcPoly(a, b, c, d, x):
    return a * x ** 3 + b * x ** 2 + c * x + d

def funcSin(x):
    return x * np.sin(2 * np.pi * x)

errorsNormal = np.random.normal(0, EPS_0, POINTS)

xSample = np.random.uniform(-1, 1, POINTS)
xFull = np.linspace(-1, 1, FULL_POINTS)

yTrue = [funcSin(x) for x in xFull]

yNormalSample = np.array([funcSin(x) + e for x, e in zip(xSample, errorsNormal)])

polyCoeffs = np.polyfit(xSample, yNormalSample, POLY_DEGREE)
polyModel = np.poly1d(polyCoeffs)

plt.figure(figsize=(8, 5))
plt.plot(xFull, yTrue, label=f"Исходная функция x*sin(2pi * x)", color='blue')
plt.scatter(xSample, yNormalSample, label="Выборка с нормальной ошибкой", color='red')
plt.plot(xFull, polyModel(xFull), label=f"Аппроксимация полиномом степени {POLY_DEGREE}", color='green')
plt.title("Полиномиальная регрессия: Оптимальное предсказание")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.ylim(-6, 6)
plt.tight_layout()
plt.savefig("./plots/polyregr_xsin2pix_optimal.png")
plt.show()