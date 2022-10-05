import random

import matplotlib.pyplot as plt
import numpy as np


k = int(input("K = "))
n = int(input("N = "))

if n % 2 or n < 2:
    raise ValueError("N must be even")

a = np.fromiter(
    (random.randint(-10, 10) for _ in range(n * n)), dtype=np.int32
).reshape(n, n)

half_n = n // 2
b = a[:half_n, :half_n]
c = a[:half_n, half_n:]
d = a[half_n:, :half_n]
e = a[half_n:, half_n:]

f = a.copy()

print(f, b, c, d, e, sep="\n\n", end="\n\n")

# количество положительных элементов в четных столбцах
x = np.sum(c[:, 1::2] > 0)
# количество отрицательных элементов в нечетных столбцах
y = np.sum(c[:, 0::2] < 0)

if x > y:
    f[:half_n, :] = f[::half_n, ::-1]
else:
    f[:half_n, half_n:], f[half_n:, half_n:] = e.copy(), c.copy()

print("После обмена", f, sep="\n\n", end="\n\n")

if np.linalg.det(a) > (np.trace(f) + np.trace(np.fliplr(f))):
    result = a * np.transpose(a) - k * f * np.linalg.inv(a)
else:
    result = (k * np.linalg.inv(a) + np.tril(a) - np.transpose(f)) * k

print("Результат", result, sep="\n\n", end="\n\n")

plt.subplot(2, 2, 1)
plt.imshow(f[:half_n, :half_n], cmap="rainbow", interpolation="bilinear")
plt.subplot(2, 2, 2)
plt.imshow(f[:half_n, half_n:], cmap="rainbow", interpolation="bilinear")
plt.subplot(2, 2, 3)
plt.imshow(f[half_n:, :half_n], cmap="rainbow", interpolation="bilinear")
plt.subplot(2, 2, 4)
plt.imshow(f[half_n:, half_n:], cmap="rainbow", interpolation="bilinear")
plt.show()

plt.subplot(2, 2, 1)
plt.plot(f[:half_n, :half_n])
plt.subplot(2, 2, 2)
plt.plot(f[:half_n, half_n:])
plt.subplot(2, 2, 3)
plt.plot(f[half_n:, :half_n])
plt.subplot(2, 2, 4)
plt.plot(f[half_n:, half_n:])
plt.show()
