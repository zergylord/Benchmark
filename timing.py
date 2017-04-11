import tensorflow as tf
import numpy as np
import time
row_dim = 100
col_dim = 100
depth = int(1e4)
steps = int(1e2)
sess = tf.Session()
A = tf.random_uniform((row_dim,col_dim))
for i in range(depth):
	A = (tf.matmul(A,A))
cur_time = time.clock()
for i in range(steps):
    sess.run(A.op)
print(time.clock()-cur_time)

cur_time = time.clock()
for i in range(steps):
    A = np.random.rand(row_dim,col_dim)
    for j in range(depth):
        A = np.matmul(A,A)
print(time.clock()-cur_time)
