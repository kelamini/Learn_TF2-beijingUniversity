import numpy as np

# 产生随机数, 因为指定 seed=1, 因此, 每次产生的随机数不变
rdm = np.random.RandomState(seed=1)
a = rdm.rand()
b = rdm.rand(2, 3)
print("a:", a)
print("b:\n", b)
