import tensorflow as tf

# 张量的四则运算
# 只有维度相同的张量才可以做四则运算

a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)

print("a:\n", a)
print("b:\n", b)

# tensor 对应元素加
add_a_b = tf.add(a, b)
print("a+b:\n", add_a_b)

# tensor 对应元素减
aubtract_a_b = tf.subtract(a, b)
print("a-b:\n", aubtract_a_b)

# tensor 对应元素乘
multiply_a_b = tf.multiply(a, b)
print("a*b:\n", multiply_a_b)

# tensor 对应元素除
divide_b_a = tf.divide(b, a)
print("b/a:\n", divide_b_a)
