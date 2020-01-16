import tensorflow.nn as nn
import tensorflow as tf
from Utils import *
#def generator_1(x):
class ST_CGAN:
    #WEIGHTS_INIT = tf.random_normal_initializer(mean=0,stddev=0.2)
    KERNEL_SIZE=3
    BATCH=1
    INPUT_SIZE=256
    IS_TRAINING=True
    def __init__(self,x,gt_mask):
        self.G1=self.generator_1(x);

        #
        sess=tf.Session()
        with tf.variable_scope('G1'):
            print('gt_mask shape')
            print(sess.run(tf.shape(gt_mask)))

        self.G1_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_mask,logits=self.G1))
        #self.G1_loss=self.L1loss(self.G1,gt_mask)

        #self.G1_loss=self.generator_loss(self.G1)
        #self.D1=self.discriminator_1(self.G1);
        #self.batch_size=1
        #self.kernel_size=3
        #return super().__init__(*args, **kwargs)

    def generator_1(self,x):
        print('making G1-network')
        sess=tf.Session()
        with tf.variable_scope('G1'):
            #Cv0
            conv0 =conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 3, 64], 1, 'g_wc0',True,False)
            print('Cv0')
            print(sess.run(tf.shape(conv0)))
            #Cv1
            conv1 = conv_layer(conv0, [self.KERNEL_SIZE, self.KERNEL_SIZE, 64, 128], 1, 'g_wc1')
            print('Cv1')
            print(sess.run(tf.shape(conv1)))
            #Cv2
            conv2 = conv_layer(conv1, [self.KERNEL_SIZE, self.KERNEL_SIZE, 128, 256], 1, 'g_wc2')
            print('Cv2')
            print(sess.run(tf.shape(conv2)))
            #Cv3
            conv3 = conv_layer(conv2, [self.KERNEL_SIZE, self.KERNEL_SIZE, 256, 512], 1, 'g_wc3')
            print('Cv3')
            print(sess.run(tf.shape(conv3)))
            #Cv4
            conv4 = conv_layer(conv3, [self.KERNEL_SIZE, self.KERNEL_SIZE, 512, 512], 1, 'g_wc4x1')
            print('Cv4')
            print(sess.run(tf.shape(conv4)))
            conv4 = conv_layer(conv4, [self.KERNEL_SIZE, self.KERNEL_SIZE, 512, 512], 1, 'g_wc4x2')
            print('Cv4')
            print(sess.run(tf.shape(conv4)))
            conv4 = conv_layer(conv4, [self.KERNEL_SIZE, self.KERNEL_SIZE, 512, 512], 1, 'g_wc4x3')
            print('Cv4')
            print(sess.run(tf.shape(conv4)))
            #Cv5
            conv5 = conv_layer(conv4, [self.KERNEL_SIZE, self.KERNEL_SIZE, 512, 512], 1, 'g_wc5',False,False)
            conv5=nn.relu(conv5)
            print('Cv5')
            print(sess.run(tf.shape(conv5)))
            #Cv6         
            conv6=deconv2d(conv5, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 512], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv6")           
            conv6=bn(conv6,self.IS_TRAINING,'g_bn6')             
            conv6=nn.relu(conv6)
            print('Cv6')
            print(sess.run(tf.shape(conv6)))
            #Cv7
            conv7=tf.concat([conv6,conv4],3)       
            conv7 =deconv2d(conv7, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 512], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv7x1")
            conv7=bn(conv7,self.IS_TRAINING,'g_bn7_x1')              
            conv7=nn.relu(conv7)
            print('Cv7')
            print(sess.run(tf.shape(conv7)))
            conv7=tf.concat([conv7,conv4],3)
            conv7 =deconv2d(conv7, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 512], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv7x2")
            conv7=bn(conv7,self.IS_TRAINING,'g_bn7_x2')              
            conv7=nn.relu(conv7)
            print('Cv7')
            print(sess.run(tf.shape(conv7)))
            conv7=tf.concat([conv7,conv4],3)
            conv7 =deconv2d(conv7, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 512], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv7x3")
            conv7=bn(conv7,self.IS_TRAINING,'g_bn7_x3')              
            conv7=nn.relu(conv7)
            print('Cv7')
            print(sess.run(tf.shape(conv7)))
            #Cv8
            conv8=tf.concat([conv7,conv3],3)          
            conv8 =deconv2d(conv8, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 256], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv8")
            conv8=bn(conv8,self.IS_TRAINING,'g_bn8')              
            conv8=nn.relu(conv8)
            print('Cv8')
            print(sess.run(tf.shape(conv8)))
            #Cv9
            conv9=tf.concat([conv8,conv2],3)
            conv9 =deconv2d(conv9, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 128], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv9")
            conv9=bn(conv9,self.IS_TRAINING,'g_bn9')              
            conv9=nn.relu(conv9)
            print('Cv9')
            print(sess.run(tf.shape(conv9)))
            #Cv10
            conv10=tf.concat([conv9,conv1],3)
            conv10 =deconv2d(conv10, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 64], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv10")
            conv10=bn(conv10,self.IS_TRAINING,'g_bn10')              
            conv10=nn.relu(conv10)
            print('Cv10')
            print(sess.run(tf.shape(conv10)))
            #Cv11
            conv11=tf.concat([conv10,conv0],3)
            conv11 =deconv2d(conv11, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 1], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv11")    
            print('Cv11')
            print(sess.run(tf.shape(conv11)))
            conv11=nn.tanh(conv11)

            return conv11

    def discriminator_1(self,x):
        print('making D1-network')
        sess=tf.Session()
        with tf.variable_scope('D1'):
        #Cv0
            conv0 = conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 4, 64], 1, 'd_wc0',True,False)
            print('Cv1')
            print(sess.run(tf.shape(conv1)))
            #Cv1 d_wc1
            conv1 = conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 64, 128], 1, 'd_wc1')
            print('Cv1')
            print(sess.run(tf.shape(conv1)))
            #Cv2
            conv2 = conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 128, 256], 1, 'd_wc2')
            print('Cv2')
            print(sess.run(tf.shape(conv2)))
            #Cv3
            conv3 = conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 256, 512], 1, 'd_wc3')
            print('Cv3')
            print(sess.run(tf.shape(conv3)))
            #Cv4
            conv4 = conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 512, 1], 1, 'd_wc4',False,False)
            print('Cv4')
            print(sess.run(tf.shape(conv4)))
            conv4=nn.sigmoid(conv3)
            return conv4
        
    # https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/    
    # https://www.datacamp.com/community/tutorials/cnn-tensorflow-python    

    #weights = {
    ## Cv0
    #'g_wc0': tf.get_variable(name="G_W0",shape=(KERNEL_SIZE, KERNEL_SIZE, 3, 64),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv1
    #'g_wc1': tf.get_variable(name="G_W1",shape=(KERNEL_SIZE, KERNEL_SIZE, 64, 128),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv2
    #'g_wc2': tf.get_variable(name="G_W2",shape=(KERNEL_SIZE, KERNEL_SIZE, 128, 256),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv3
    #'g_wc3': tf.get_variable(name="G_W3",shape=(KERNEL_SIZE, KERNEL_SIZE, 256, 512),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv4
    #'g_wc4': tf.get_variable(name="G_W4",shape=(KERNEL_SIZE, KERNEL_SIZE, 512, 512),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv5
    #'g_wc5': tf.get_variable(name="G_W5",shape=(KERNEL_SIZE, KERNEL_SIZE, 512, 512),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv6
    #'g_wc6': tf.get_variable(name="G_W6",shape=(KERNEL_SIZE, KERNEL_SIZE, 512, 512),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv7
    #'g_wc7': tf.get_variable(name="G_W7",shape=(KERNEL_SIZE, KERNEL_SIZE, 1024, 512),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv8
    #'g_wc8': tf.get_variable(name="G_W8",shape=(KERNEL_SIZE, KERNEL_SIZE, 1024, 256),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv9
    #'g_wc9': tf.get_variable(name="G_W9",shape=(KERNEL_SIZE, KERNEL_SIZE, 512, 128),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv10
    #'g_wc10': tf.get_variable(name="G_W10",shape=(KERNEL_SIZE, KERNEL_SIZE, 256, 64),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    ## Cv11
    #'g_wc11': tf.get_variable(name="G_W11",shape=(KERNEL_SIZE, KERNEL_SIZE, 128, 1),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    # # Discriminator Cv0
    #'d_wc0': tf.get_variable(name="D_W0",shape=(KERNEL_SIZE, KERNEL_SIZE, 4, 64),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    # # Discriminator Cv1
    #'d_wc1': tf.get_variable(name="D_W1",shape=(KERNEL_SIZE, KERNEL_SIZE, 64, 128),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    # # Discriminator Cv2
    #'d_wc2': tf.get_variable(name="D_W2",shape=(KERNEL_SIZE, KERNEL_SIZE, 128, 256),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    # # Discriminator Cv3
    #'d_wc3': tf.get_variable(name="D_W3",shape=(KERNEL_SIZE, KERNEL_SIZE, 256, 512),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32),
    # # Discriminator Cv4
    #'d_wc4': tf.get_variable(name="D_W4",shape=(KERNEL_SIZE, KERNEL_SIZE, 512, 1),initializer = tf.random_normal_initializer(mean=0,stddev=0.2), dtype = tf.float32)
    #}

    #biases = {
    #'g_bc0': tf.get_variable('G_B0', shape=(64), initializer = tf.random_normal_initializer(mean=0,stddev=0.2)),
    #'g_bc1': tf.get_variable('G_B1', shape=(128), initializer = tf.random_normal_initializer(mean=0,stddev=0.2)),
    #'g_bc2': tf.get_variable('G_B2', shape=(256),  initializer = tf.random_normal_initializer(mean=0,stddev=0.2)),
    #'g_bc3': tf.get_variable('G_B3', shape=(512),  initializer = tf.random_normal_initializer(mean=0,stddev=0.2)),
    #'g_bc4': tf.get_variable('G_B4', shape=(512),  initializer = tf.random_normal_initializer(mean=0,stddev=0.2)),
    #'g_bc4_x2': tf.get_variable('G_B4_x2', shape=(512),  initializer = tf.random_normal_initializer(mean=0,stddev=0.2)),
    #'g_bc4_x3': tf.get_variable('G_B4_x3', shape=(512),  initializer = tf.random_normal_initializer(mean=0,stddev=0.2)),
    #'g_bc5': tf.get_variable('G_B5', shape=(512),  initializer = tf.random_normal_initializer(mean=0,stddev=0.2))

    #}