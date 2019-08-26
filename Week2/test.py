import tensorflow as tf
import numpy as np

a = tf.constant(np.array([[1,2], [3,4]]))
b = tf.constant(np.array([1,2]))

c = tf.tensordot(a,b,axes=1)

with tf.Session() as sess:
    print(sess.run(c))
tf.convert_to_tensor