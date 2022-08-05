import tensorflow as tf

x = tf.Variable(4) # 定义可训练变量
x.assign_sub(1) # x 自减 1

print("x:", x)  # 4 - 1 = 3
