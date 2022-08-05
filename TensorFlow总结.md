## TensorFlow 基础
### 常用函数
#### 1.  tf.constant(张量内容, dtype=数据类型(可选))
解释：创建常量 tensor，数据类型可选：tf.int32；tf.int64；tf.float32；float64。
代码：

```python
# constant.py

import tensorflow as tf

a = tf.constant([1, 5], dtype=tf.int64)
print("a: ", a)
print("a.dtype: ", a.dtype)
print("a.shape: ", a.shape)
```
输出：
```output
a: tf.Tensor([1 5], shape=(2,), dtype=int64)
a.dtype: <dtype: 'int64'>
a.shape: (2,)
```
#### 2. tf. convert_to_tensor(数据名, dtype=数据类型(可选))
解释：将 numpy 格式化为 Tensor 格式。
代码：
```python
# convert_to_tensor.py

import tensorflow as tf
import numpy as np

a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print("a: ", a)
print("b: ", b)
```
输出：
```output
a: [0 1 2 3 4]
b: tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)
```
#### 3. tf. zeros(维度) & tf.ones(维度) & tf. fill(维度, 指定值)
解释：创建全 0 张量；创建全 1 张量；创建指定值张量。
代码：
```python
# zeros_ones_fill.py

import tensorflow as tf

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print("a: ", a)
print("b: ", b)
print("c: ", c)
```
输出：
```output
a: tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
b: tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
c: tf.Tensor(
[[9 9]
 [9 9]], shape=(2, 2), dtype=int32)
```
#### 4. tf. random.normal(维度 , mean=均值, stddev=标准差)
解释：生成正态分布的随机数，默认均值为 0，标准差为 1。
代码：

```python
# random_normal.py

import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d: ", d)
```
输出：
```output
d: tf.Tensor(
[[ 1.0898712   0.50711805]
 [ 0.14399704 -0.98084044]], shape=(2, 2), dtype=float32)
```
#### 5. tf. random.truncated_normal(维度, mean=均值, stddev=标准差)
解释：生成截断式正态分布的随机数，能使生成的这些随机数更集中一些，如果随机生成数据的取值在 (µ - 2σ, u + 2σ ) 之外则重新进行生成，保证了生成值在均值附近。
代码：

```python
# random_truncated_normal.py

import tensorflow as tf

e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e: ", e)
```
输出：
```output
e: tf.Tensor(
[[-0.2508564   1.7363298 ]
 [ 0.48450977  1.5890658 ]], shape=(2, 2), dtype=float32)
```
#### 6. tf.random. uniform(维度, minval=最小值, maxval=最大值)
解释：生成指定维度的均匀分布随机数，用 minval 给定随机数的最小值，用 maxval 给定随机数的最大值，最小、最大值是前闭后开区间。
代码：
```python
# random.uniform.py

import tensorflow as tf

f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f: ", f)
```
输出：
```output
f: tf.Tensor(
[[0.33746672 0.17689085]
 [0.087111   0.7979419 ]], shape=(2, 2), dtype=float32)
```
#### 7. tf.cast(张量名, dtype=数据类型)
解释：强制将 Tensor 转换为该数据类型。
代码：

```python
# cast.py

import tensorflow as tf

x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1: ", x1)
x2 = tf.cast(x1, tf.int32)
print("x2: ", x2)
```
输出：
```output
x1: tf.Tensor([1. 2. 3.], shape=(3,), dtype=float64)
x2: tf.Tensor([1 2 3], shape=(3,), dtype=int32)
```
#### 8. tf.reduce_min(张量名) & tf.reduce_max(张量名)
解释：计算张量维度上元素的最小最大值。
代码：

```python
# reduce_minmax.py

import tensorflow as tf

x1 = tf.constant([1, 2, 3])
reduce_min = tf.reduce_min(x1)
print("minimum of x1: ", reduce_min)
reduce_max = tf.reduce_max(x1)
print("maxmum of x1: ", reduce_max)
```
输出：
```output
minmum of x1: tf.Tensor(1, shape=(), dtype=int32)
maxmum of x1: tf.Tensor(3, shape=(), dtype=int32)
```
#### 9. tf.reduce_mean(张量名, axis=操作轴)
解释：计算张量沿着指定维度的平均值。axis=0 表示纵向操作；axis=1 表示横向操作；如不指定 axis，则表示对所有元素进行操作。
代码：
```python
# reduce_mean.py

import tensorflow as tf

m = tf.constant([[1, 2, 3], [2, 2, 3]])
print("m: ", m)
reduce_mean_y = tf.reduce_mean(m, axis=0)
print("mean of m_y: ", reduce_mean_y)
reduce_mean_x = tf.reduce_mean(m, axis=1)
print("mean of m_x: ", reduce_mean_x)
reduce_mean = tf.reduce_mean(m)
print("mean of m: ", reduce_mean)
```
输出：
```output
m: [[1 2 3]
   [2 2 3]], shape=(2, 3), dtype=int32)
mean of m_y: tf.Tensor([1 2 3], shape=(3,), dtype=int32)
mean of m_x: tf.Tensor([2 2], shape=(2,), dtype=int32)
mean of m: tf.Tensor(2, shape=(), dtype=int32)
```
#### 10. tf.reduce_sum (张量名, axis=操作轴)
解释：计算张量沿着指定维度的和，如不指定 axis，则表示对所有元素进行操作。
代码：

```python
# reduce_mean.py

import tensorflow as tf

m = tf.constant([[1, 2, 3], [2, 2, 3]])
print("m:", m)
reduce_sum_y = tf.reduce_sum(m, axis=0)
print("sum of m_y:", reduce_sum_y)
reduce_sum_x = tf.reduce_sum(m, axis=1)
print("sum of m_x: ", reduce_sum_x)
reduce_sum = tf.reduce_sum(m)
print("sum of m: ", reduce_sum)
```
输出：
```output
m: [[1 2 3]
  [2 2 3]], shape=(2, 3), dtype=int32)
sum of m_y: tf.Tensor([3 4 6], shape=(3,), dtype=int32)
sum of m_x: tf.Tensor([6 7], shape=(2,), dtype=int32)
sum of m: tf.Tensor(13, shape=(), dtype=int32)
```
#### 11. tf.Variable(initial_value, trainable, validate_shape,name)
解释：将变量标记为 “可训练” 的，被它标记了的变量，会在反向传播中记录自己的梯度信息。
**initial_value:** 默认为 None，可以搭配 tensorflow 随机生成函数来初始化参数；
**trainable:** 默认为 True，表示可以后期被算法优化的，如果不想该变量被优化，即改为 False；
**validate_shape:** 默认为 True，形状不接受更改，如果需要更改，validate_shape=False；
**name:** 默认为 None，给变量确定名称。
代码：

```python
import rensorflow as tf

w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
```
#### 12.  tf.add(张量 1, 张量2) & tf.subtract(张量 1, 张量 2) & tf.multiply(张量 1, 张量 2) & tf.divide(张量 1, 张量 2)
解释：张量的四则运算。
代码：

```python
# add_sub_mul_div.py

import tensorflow as tf

a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)

print("a: ", a)
print("b: ", b)
add_a_b = tf.add(a, b)  # 对应元素加
print("a+b: ", add_a_b)
aubtract_a_b = tf.subtract(a, b)  # 对应元素减
print("a-b: ", aubtract_a_b)
multiply_a_b = tf.multiply(a, b)  # 对应元素乘
print("a*b: ", multiply_a_b)
divide_b_a = tf.divide(b, a)  # 对应元素除
print("b/a: ", divide_b_a)
```
输出：
```output
a: tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
b: tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
a+b: tf.Tensor([[4. 4. 4.]], shape=(1, 3), dtype=float32)
a-b: tf.Tensor([[-2. -2. -2.]], shape=(1, 3), dtype=float32)
a*b: tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
b/a: tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
```
#### 13. tf.square(张量名) & tf.pow(张量名, n 次方数) & tf.sqrt (张量名)
解释：幂方运算。（二次方；n 次方；开方）
代码：
```python
# square_pow_sqrt.py

import tensorflow as tf

a = tf.fill([1, 2], 3.)
print("a: ", a)

square_a = tf.square(a)  # 计算张量 a 的平方
print("a的平方: ", square_a)
pow_a = tf.pow(a, 3)  # 计算张量 a 的三次幂
print("a的三次幂: ", pow_a)
sqrt_a = tf.sqrt(a)  # 计算张量 a 的开方
print("a的开方: ", sqrt_a)
```
输出：
```output
a: tf.Tensor([[3. 3.]], shape=(1, 2), dtype=float32)
a的平方: tf.Tensor([[9. 9.]], shape=(1, 2), dtype=float32)
a的三次幂: tf.Tensor([[27. 27.]], shape=(1, 2), dtype=float32)
a的开方: tf.Tensor([[1.7320508 1.7320508]], shape=(1, 2), dtype=float32)
```
#### 14. tf.matmul(矩阵 1, 矩阵 2)
解释：实现两个矩阵的相乘。
代码：

```python
# matmul.py

import tensorflow as tf

a = tf.ones([3, 2])
b = tf.fill([2, 3], 3.)
print("a: ", a)
print("b: ", b)
matmul_a_b = tf.matmul(a, b)
print("a×b: ", matmul_a_b)
```
输出：
```output
a: tf.Tensor(
[[1. 1.]
 [1. 1.]
 [1. 1.]], shape=(3, 2), dtype=float32)
b: tf.Tensor(
[[3. 3. 3.]
 [3. 3. 3.]], shape=(2, 3), dtype=float32)
a×b: tf.Tensor(
[[6. 6. 6.]
 [6. 6. 6.]
 [6. 6. 6.]], shape=(3, 3), dtype=float32)
```
#### 15. tf.data.Dataset.from_tensor_slices((输入特征, 标签))
解释：切分传入张量的第一维度，生成 输入特征/标签 对。
构建数据集，此函数对 Tensor 格式与 Numpy 格式均适用，其切分的是第一维度，表征数据集中数据的数量，之后切分 batch 等操作都以第一维为基础。
代码：
```python
# from_tensor_slices.py

import tensorflow as tf

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print('element: ', element)
```
输出：
```output
element: (<tf.Tensor: shape=(), dtype=int32, numpy=12>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
element: (<tf.Tensor: shape=(), dtype=int32, numpy=23>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
element: (<tf.Tensor: shape=(), dtype=int32, numpy=10>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
element: (<tf.Tensor: shape=(), dtype=int32, numpy=17>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
```
#### 16. tf.GradientTape( )
解释：搭配 with 结构计算损失函数在某一张量处的梯度。
代码：
```python
# GradientTape.py

import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))  # 定义可训练的变量 x
    y = tf.pow(x, 2)  # 定义损失函数（运算模型）
grad = tape.gradient(y, x)
print('grad: ', grad)
```
输出：
```output
grad: tf.Tensor(6.0, shape=(), dtype=float32)
```
#### 17. enumerate(列表名)
解释：枚举出每一个元素，并在元素前配上对应的索引号，常在 for 循环中使用。
代码：
```python
# enumerate.py

seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)
```
输出：
```output
0 one
1 two
2 three
```
#### 18. tf.one_hot(待转换数据, depth=几分类)
解释：实现用独热码表示标签，在分类问题中很常见。标记类别为为 1 和 0，其中 1 表示是， 0 表示非。
代码：
```python
# one_hot.py

import tensorflow as tf

classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为 0，最大为 2
output = tf.one_hot(labels, depth=classes)  # 独热编码
print("result of labels: ", output)
```
输出：
```output
result of labels: tf.Tensor(
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]], shape=(3, 3), dtype=float32)
```
#### 19.  tf.nn.softmax( )
解释：使前向传播的输出值符合概率分布。
代码：
```python
# softmax.py

import tensorflow as tf

y = tf.constant([1.01, 2.01, -0.66])

y_pro = tf.nn.softmax(y)
print("After softmax, y_pro is: ", y_pro)
print("The sum of y_pro: ", tf.reduce_sum(y_pro))
```
输出：
```output
After softmax, y_pro is: tf.Tensor([0.25598174 0.69583046 0.04818781], shape=(3,), dtype=float32)
The sum of y_pro: tf.Tensor(1.0, shape=(), dtype=float32)
```
#### 20. squeeze()
解释：默认删除所有维度是 1 的维度。如果不想删除所有维度为 1 的维度，可以通过指定 axis 来删除特定维度为 1 的维度。
代码：

```python
import tensorflow as tf

x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])  # 5.8,4.0,1.2,0.2（0）
w1 = tf.constant([[-0.8, -0.34, -1.4],
                  [0.6, 1.3, 0.25],
                  [0.5, 1.45, 0.9],
                  [0.65, 0.7, -1.2]])
b1 = tf.constant([2.52, -3.1, 5.62])

y = tf.matmul(x1, w1) + b1

print("x1.shape: ", x1.shape)
print("w1.shape: ", w1.shape)
print("b1.shape: ", b1.shape)
print("y.shape: ", y.shape)
print("y: ", y)

y_dim = tf.squeeze(y)
y_pro = tf.nn.softmax(y_dim)
print("y_dim: ", y_dim)
print("y_pro: ", y_pro)
```
输出：
```output
x1.shape: (1, 4)
w1.shape: (4, 3)
b1.shape: (3,)
y.shape: (1, 3)
y: tf.Tensor([[ 1.0099998  2.008  -0.65999985]], shape=(1, 3), dtype=float32)
y_dim: tf.Tensor([ 1.0099998  2.008  -0.65999985], shape=(3,), dtype=float32)
y_pro: tf.Tensor([0.2563381  0.69540703  0.04825491], shape=(3,), dtype=float32)
```
#### 21. assign_sub()
解释：对参数实现自更新。使用此函数前需利用 tf.Variable 定义变量 w 为可训练。
代码：
```python
# assign_sub.py

import tensorflow as tf

x = tf.Variable(4)  # 定义可训练变量
x.assign_sub(1)  # x 自减 1
print("x: ", x)  # 4 - 1 = 3
```
输出：
```output
x: <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>
```
#### 22.  tf.argmax (张量名, axis=操作轴)
解释：返回张量沿指定维度最大值的索引。
代码：
```python
# argmax.py

import numpy as np
import tensorflow as tf

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print("test: ", test)
print("每一列的最大值的索引: ", tf.argmax(test, axis=0))
print("每一行的最大值的索引: ", tf.argmax(test, axis=1))
```
输出：
```output
test: [[1 2 3]
       [2 3 4]
       [5 4 3]
       [8 7 2]]
每一列的最大值的索引: tf.Tensor([3 3 1], shape=(3,), dtype=int64)
每一行的最大值的索引: tf.Tensor([2 2 0 0], shape=(4,), dtype=int64)
```





