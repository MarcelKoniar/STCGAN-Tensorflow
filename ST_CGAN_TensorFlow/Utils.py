import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import tensorflow.nn as nn

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        #biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        biases = tf.get_variable('biases', [output_shape[-1]], initializer = tf.random_normal_initializer(mean=0,stddev=0.2))

        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def conv_layer(x, filtershape, stride, name,LReLU=True,BN=True,IS_Training=True):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name = 'weight',
            shape = filtershape,
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.2),
            trainable = True)
        conv = tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding= 'SAME')
        conv_biases = tf.Variable(tf.constant(0.1, shape = [filtershape[3]], dtype = tf.float32),
                                trainable=True, name ='bias')
        bias = tf.nn.bias_add(conv, conv_biases)       
        output=bias
        if(BN==True):
            output=bn(output,IS_Training,name+'_bn')
        if(LReLU==True):
            output=nn.leaky_relu(bias)
       
        #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
        #img_filt = tf.reshape(filters[:,:,:,1], [-1,filtershape[0],filtershape[1],1])
        #tf.summary.image('conv_filter',img_filt)
        return output


#def binary_crossentropy(output, target, from_logits=False):
#  """Binary crossentropy between an output tensor and a target tensor.
#  Arguments:
#      output: A tensor.
#      target: A tensor with the same shape as `output`.
#      from_logits: Whether `output` is expected to be a logits tensor.
#          By default, we consider that `output`
#          encodes a probability distribution.
#  Returns:
#      A tensor.
#  """
#  # Note: nn.softmax_cross_entropy_with_logits
#  # expects logits, Keras expects probabilities.
#  if not from_logits:
#    # transform back to logits
#    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
#    output = clip_ops.clip_by_value(output, epsilon, 1 - epsilon)
#    output = math_ops.log(output / (1 - output))
#  return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

def image_show(np_image):
    img = Image.fromarray(np_image,'RGB')
    img.show()


def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)