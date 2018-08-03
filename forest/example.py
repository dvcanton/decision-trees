import numpy as np
import matplotlib.pyplot as plt

# Some fake datasets
x = np.random.uniform(1, 100, 1000)
y = np.log(x) + np.random.normal(0, 0.3, 1000)

print("First 5 elements: " + np.array_str(y[:5],precision=2))

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['toolbar'] = 'None'

plt.title("Basic log function")
plt.scatter(x, y, s=2, marker='o', label="l(x) with noise")
plt.plot(np.arange(1, 100), np.log(np.arange(1, 100)), label="log(x)")

plt.xlabel("x")
plt.ylabel("y")

plt.legend(loc="best")
plt.show()
