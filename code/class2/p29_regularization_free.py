# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据/标签 生成x_train, y_train
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1,2)
y_train = np.vstack(y_data).reshape(-1,1)

Y_c = [['red' if y else 'blue'] for y in y_train]

# 转换 x 的数据类型, 否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices 函数切分传入的张量的第一个维度, 生成相应的数据集, 使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数, 输入层为 2 个神经元, 隐藏层为 11 个神经元, 1 层隐藏层, 输出层为 1 个神经元
# 用tf.Variable() 保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01  # 学习率
epoch = 400  # 循环轮数

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_train - y))

        # 计算 loss 对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        # w1 = w1 - lr * w1_grad tape.gradient 是自动求导结果与 [w1, b1, w2, b2], 索引为0, 1, 2, 3
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每 20 个 epoch，打印 loss 信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*****************predict*****************")
# xx 在 -3 到 3 之间以步长为 0.01, yy 在 -3 到 3 之间以步长 0.01, 生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将 xx , yy 拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络, 进行预测, probs 为输出
probs = []
for x_test in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y 为预测结果
    probs.append(y)

# 取第 0 列给 x1, 取第 1 列给 x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs 的 shape 调整成 xx 的样子
probs = np.array(probs).reshape(xx.shape)
# squeeze 去掉纬度是 1 的纬度, 相当于去掉 [['red'],[''blue]], 内层括号变为 ['red','blue']
plt.scatter(x1, x2, color=np.squeeze(Y_c))
# 把坐标 xx, yy 和对应的值 probs 放入 contour 函数
# 连接 probs 值为 0.5 的所有点
# plt.show 后, 显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 读入红蓝点, 画出分割线, 不包含正则化
# 不清楚的数据 ,建议 print 出来查看