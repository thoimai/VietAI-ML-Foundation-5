import numpy as np
import pandas as pd
import tensorflow as tf

#Read data
data_df = pd.read_excel('cobbdouglas_edit.xls')
# print(data_df[4:10])

P_data = data_df['Relative Capital Stock, 1899=100'].values
L_data = data_df['Relative Number of Workers, 1899=100'].values
K_data = data_df['Index of Manufactures'].values

L_data = np.asarray(L_data)
K_data = np.asarray(K_data)
P_data = np.asarray(P_data)

n_samples = P_data.shape[0]
learning_rate = 0.005
# training_epochs = 10000
# display_step = 1000
# batch_size = n_samples

y = tf.placeholder(tf.float64, name='K')
L = tf.placeholder(tf.float64, name='L')
K = tf.placeholder(tf.float64, name='y')

b = tf.Variable(np.random.random(), dtype=tf.float64)
alpha = tf.Variable(np.random.random(), dtype=tf.float64)

y_predict = tf.multiply(b, (tf.pow(L, alpha)*tf.pow(K, 1-alpha)))

loss = tf.square(y_predict - y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        total_loss = 0
        for l,k,y in zip(L_data, K_data, P_data):
            _, _loss = sess.run([optimizer,loss], feed_dict={L:l, K:k, y:y})
            total_loss += _loss
        if(i+1) %10 == 0:
            print('Epoch {0}: {1}'.format(i+1, total_loss/n_samples))



