import tensorflow as tf

m = tf.constant([[1, 2, 3], [2, 2, 3]])
print("m:\n", m)

# axis=0 表示纵向操作(沿经度方向)
# axis=1 表示横向操作(沿纬度方向)

# 计算张量沿着指定维度的平均值，如不指定 axis，则表示对所有元素进行操作
reduce_mean = tf.reduce_mean(m, axis=1)
print("mean of m:\n", reduce_mean)  # 求每一行的均值

# 计算张量沿着指定维度的和，如不指定 axis，则表示对所有元素进行操作
reduce_sum = tf.reduce_sum(m, axis=1)
print("sum of m:\n", reduce_sum)  # 求每一行的和
