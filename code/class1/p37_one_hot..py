import tensorflow as tf

classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为0，最大为2

# 独热编码
output = tf.one_hot(labels, depth=classes)
print("result of labels1:\n", output)
