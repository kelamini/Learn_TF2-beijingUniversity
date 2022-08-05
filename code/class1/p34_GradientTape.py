import tensorflow as tf

# 计算损失函数在某一张量处的梯度
with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))  # 定义可训练的变量 x
    y = tf.pow(x, 2)  # 定义损失函数（运算模型）
grad = tape.gradient(y, x)
print('grad:\n', grad)
