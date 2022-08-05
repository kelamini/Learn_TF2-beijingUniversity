import tensorflow as tf

# 创建指定维度的均匀分布随机 tensor
# 用 minval 给定随机数的最小值 ，用 maxval 给定随机数的最大值
# 最小、最大值是前闭后开区间
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f:", f)
