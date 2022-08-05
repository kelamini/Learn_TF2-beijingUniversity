import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))

epoch = 100
LR_BASE = 0.2  # 最初学习率
LR_DECAY = 0.99  # 学习率衰减率
LR_STEP = 4  # 喂入多少轮BATCH_SIZE后，更新一次学习率

# for epoch 定义顶层循环，表示对数据集循环 epoch 次
# 此例数据集数据仅有 1 个 w, 初始化时候 constant 赋值为 5, 循环 100 次迭代。
for epoch in range(epoch):
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient 函数告知谁对谁求导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch, w is %f, loss is %f, lr is %f" % (epoch, w.numpy(), loss, lr))
