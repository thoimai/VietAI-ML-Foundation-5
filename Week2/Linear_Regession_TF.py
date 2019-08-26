import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils_function import load_Boston_housing_data
import numpy as np

train_X, test_X, train_Y, test_Y = load_Boston_housing_data(feature_ind = [2,5])

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

learning_rate = 0.005
training_epochs = 10000
display_step = 1000
n_samples, dimension = train_X.shape
batch_size = n_samples # Full Batch Gradient Descent

#TODO: implement input and parameter for tensorflow.
train_X = tf.constant(train_X, dtype=tf.float64)

train_Y = tf.reshape(tensor=train_Y, shape=(-1, 1))
train_Y = tf.convert_to_tensor(train_Y) # convert train_Y to tensor tf

# Set model weights
W = tf.Variable(np.zeros((dimension,1))) # create weights variable to train. size=(dimension, 1)
b = tfe.Variable(np.random.normal(size=(1, 1)), trainable=True)
# print(W)
# print(b)

#TODO: implement a linear regression function
def tf_lr_hypothesis(X, W, b):
    return tf.add(tf.matmul(X,W), b)

#TODO: implement a cost function
def tf_mse_cost(Y_hat, Y):
    return tf.reduce_sum(tf.square(Y_hat - Y)/(2*n_samples))

# TODO: implemement GD
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        Y_hat = tf_lr_hypothesis(train_X, W, b) # apply linear regression function here
        mse_cost = tf_mse_cost(Y_hat, train_Y) # apply mse cost here.
    grads = tape.gradient(mse_cost, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))
    if (epoch + 1) % display_step == 0:
        print("Epoch:", epoch + 1, "| Cost:", mse_cost.numpy())

