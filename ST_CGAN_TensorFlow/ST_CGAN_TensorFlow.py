import tensorflow as tf
import time

tf.set_random_seed(42)
A = tf.random_normal([10000,10000])
B = tf.random_normal([10000,10000])

def check():
    start_time = time.time()
    with tf.Session() as sess:
        print(sess.run(tf.reduce_sum(tf.matmul(A,B))))
    print("It took {} seconds".format(time.time() - start_time))

#check()
sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 8, 8, 128]
strides = [1, 2, 2, 1]

l = tf.constant(0.1, shape=[1, 32, 32, 4])
w = tf.constant(0.1, shape=[7, 7, 128, 4])

h1 = tf.nn.conv2d_transpose(l, w, output_shape=output_shape, strides=strides, padding='SAME')
#print(sess.run(h1))
print(sess.run(tf.shape(h1)))
