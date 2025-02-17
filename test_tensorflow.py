import tensorflow as tf

# Simple test to check TensorFlow functionality
print("TensorFlow version:", tf.__version__)

# Create a simple tensor
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Perform a matrix multiplication
c = tf.matmul(a, b)
print("Matrix multiplication result:\n", c.numpy()) 