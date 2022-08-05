import tensorflow as tf

a = tf.fill([1, 2], 3.)
print("a:\n", a)

# 计算张量 a 的平方
square_a = tf.square(a)
print("a的平方:\n", square_a)

# 计算张量 a 的三次幂
pow_a = tf.pow(a, 3)
print("a的三次幂:\n", pow_a)

# 计算张量 a 的开方
sqrt_a = tf.sqrt(a)
print("a的开方:\n", sqrt_a)
