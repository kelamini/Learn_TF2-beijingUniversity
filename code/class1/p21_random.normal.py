import tensorflow as tf

# 创建正态分布的随机 tensor ，默认均值为 0.5 ，标准差为 1
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)

# 创建截断式正态分布的随机 tensor
# 能使生成的这些随机数更集中一些，如果随机生成数据的取值
# 在(µ - 2σ， u + 2σ ) 之外则重新进行生成，保证了生成值在均值附近
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)
