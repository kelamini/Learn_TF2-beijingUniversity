import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# 沿竖直方向堆叠矩阵
c = np.vstack((a, b))
print("c:\n", c)
# 沿水平方向堆叠矩阵
d = np.hstack((a, b))
print("d:\n", d)
