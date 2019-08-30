import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

data_df = pd.read_excel('cobbdouglas.xls', header=None, skiprows=1)
# print(data_df)

data = np.asarray(data_df)
# print(data)

K = data_df[1].values
L = data_df[2].values
P = data_df[3].values
# print(P)
# print(L)
# print(K)

X = np.log(L) - np.log(K)
y = np.log(P) - np.log(K)
# print(X,y)

N = X.shape[0]
# print(N)

learning_rate = 0.005
training_epochs = 1000
display_step = 100
n_samples = X.shape[0]
batch_size = n_samples

train_X = tf.placeholder(dtype=tf.float32)
train_y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(np.random.random())
b = tf.Variable(np.random.random())
print(W)
print(b)

y_predict = W*train_X + b
loss = tf.square(y_predict - train_y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    for i in range(training_epochs):
        total_loss = 0
        for x, j in zip(X, y):
            # Session excute optimizer and fetch values of loss
            _, _loss = sess.run([optimizer, loss], feed_dict={train_X: x, train_y: j})
            total_loss += _loss
        if (i + 1) % display_step == 0:
            print('Epoch {0}: {1}'.format(i + 1, total_loss / n_samples))

    W_out, b_out = sess.run([W, b])
    print('alpha = ', W_out)
    print('log(b) = ', b_out)