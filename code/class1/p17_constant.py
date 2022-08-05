import tensorflow as tf

# 创建 tensor
a = tf.constant([1, 5], dtype=tf.int64)

print("a:\n", a)
print("a.dtype:\n", a.dtype)
print("a.shape:\n", a.shape)

# 本机默认 tf.int32  可去掉dtype试一下 查看默认值
