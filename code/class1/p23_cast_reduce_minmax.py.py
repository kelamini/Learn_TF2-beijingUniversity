import tensorflow as tf

x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1:", x1)

# 强制类型转换
x2 = tf.cast(x1, tf.int32)
print("x2:", x2)

# 计算张量维度上元素的最小值
reduce_min = tf.reduce_min(x2)
print("minimum of x2:", reduce_min)

# 计算张量维度上元素的最大值
reduce_max = tf.reduce_max(x2)
print("maxmum of x2:", reduce_max)
