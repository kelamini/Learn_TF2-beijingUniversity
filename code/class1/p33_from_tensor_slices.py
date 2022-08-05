import tensorflow as tf

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])

# 切分传入张量的第一维度，生成 输入特征/标签 对
# 构建数据集，此函数对 Tensor 格式与 Numpy 格式均适用
# 其切分的是第一维度，表征数据集中数据的数量
# 之后切分 batch 等操作都以第一维为基础
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print('element:\n', element)
