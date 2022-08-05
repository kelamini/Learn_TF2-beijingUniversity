import tensorflow as tf

a = tf.ones([3, 2])
b = tf.fill([2, 3], 3.)
print("a:\n", a)
print("b:\n", b)

# 计算张量 a 与张量 b 的矩阵乘
matmul_a_b = tf.matmul(a, b)
print("a*b:", matmul_a_b)
