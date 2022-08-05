# 导入所需模块
import tensorflow as tf
from sklearn import datasets
import numpy as np

# 准备数据集
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 搭建网络模型结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

# 配置网络模型训练方法
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 执行网络模型训练过程
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# 打印网络模型结构
model.summary()
