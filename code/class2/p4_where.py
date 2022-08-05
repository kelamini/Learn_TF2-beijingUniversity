import tensorflow as tf
###############################################################
# tf.greater(a,b)
# 功能: 比较 a、b 两个值的大小
# 返回值: 一个列表，元素值都是 true 和 false
###############################################################
# tf.where(condition, x, y)
# condition, x, y 相同维度, condition 是 bool 型值, True/False
# 返回值是对应元素
# condition中元素为True的元素替换为x中的元素
# 为False的元素替换为y中对应元素
###############################################################
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
# 若 a > b ，返回 a 对应位置的元素，否则返回 b 对应位置的元素
c = tf.where(tf.greater(a, b), a, b)
print("c:\n", c)


