import tensorflow as tf
import numpy as np

# numpy 转 tensor
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)

print("a:", a)
print("b:", b)
