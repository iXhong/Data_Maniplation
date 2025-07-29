import numpy as np
import matplotlib.pyplot as plt

true_length = 100.0  # 真正的绳子长度
sigma = 0.5          # 测量误差标准差
n = 100              # 测量次数

measurements = np.random.normal(loc=true_length, scale=sigma, size=n)

print(type(measurements))

print(f"mean: {np.mean(measurements):.3f}")
print(f"standard variance: {np.std(measurements, ddof=1):.3f}")

plt.hist(measurements, bins=20, alpha=0.7, edgecolor='black')
plt.axvline(true_length, color='red', linestyle='--', label='truth')
plt.title("distribution")
plt.xlabel("measure length")
plt.ylabel("freq")
plt.legend()
plt.show()
