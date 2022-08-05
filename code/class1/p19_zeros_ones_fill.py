import tensorflow as tf

# 创建全零 tensor
a = tf.zeros([2, 3])

# 创建全 1 tensor
b = tf.ones(4)

# 创建指定值填充 tensor
c = tf.fill([2, 2], 9)

print("a:", a)
print("b:", b)
print("c:", c)
