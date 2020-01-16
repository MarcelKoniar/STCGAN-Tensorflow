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
    def __init__(self,shadow_input,gt_mask,gt_shadowfree):
        self.G1=self.generator_1(shadow_input);
        self.G1_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_mask,logits=self.G1))
        self.D1_fake=self.discriminator_1(tf.concat([shadow_input,self.G1],3),'D1_fake')
        self.D1_real=self.discriminator_1(tf.concat([shadow_input,gt_mask],3),'D1_real')
        self.D1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_fake, labels=self.D1_real))
        
        #self.G2=self.generator_2(tf.concat([shadow_input,gt_mask],3))
        self.G2=self.generator_2(tf.concat([shadow_input,self.G1],3))
        ##todo G2 loss
        self.G2_loss=tf.reduce_mean(tf.abs(gt_shadowfree-self.G2))

        self.D2_fake=self.discriminator_2(tf.concat([shadow_input,self.G1,self.G2],3),'D2_fake')
        self.D2_real=self.discriminator_2(tf.concat([shadow_input,gt_mask,gt_shadowfree],3),'D2_real')
        ##todo D2 loss
        self.D2_loss=tf.reduce_mean(tf.abs(self.D2_real-self.D2_fake))

        ##testing g loss with d loss
        #self.G1_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_mask,logits=self.G1))
        #self.D1_fake_input=self.discriminator_1(tf.concat([shadow_input,self.G1],3),'D1_fake')
        #self.D1_real_input=self.discriminator_1(tf.concat([shadow_input,gt_mask],3),'D1_real')
        #self.D1_fake_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_fake,labels=tf.zeros_like(self.D1_fake_input)))
        #self.D1_real_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_real,labels=tf.ones_like(self.D1_real_input)))
        #self.D1_loss = self.D1_fake_loss + self.D1_real_loss
        
        ##self.G2=self.generator_2(tf.concat([shadow_input,gt_mask],3))
        #self.G2=self.generator_2(tf.concat([shadow_input,self.G1],3))
        ###todo G2 loss
        #self.G2_loss=tf.reduce_mean(tf.abs(gt_shadowfree-self.G2))

        #self.D2_fake=self.discriminator_2(tf.concat([shadow_input,self.G1,self.G2],3),'D2_fake')
        #self.D2_real=self.discriminator_2(tf.concat([shadow_input,gt_mask,gt_shadowfree],3),'D2_real')
        ###todo D2 loss
        #self.D2_loss=tf.reduce_mean(tf.abs(self.D2_real-self.D2_fake))       


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
            #conv11=nn.sigmoid(conv11)

            return conv11    

    def discriminator_1(self,x,name):
        print('making D1-network')
        sess=tf.Session()
        with tf.variable_scope('D1'):
            #Cv0
            conv0 = conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 4, 64], 1, str(name +'d_wc0'),True,False)
            print('Cv1')
            print(sess.run(tf.shape(conv0)))
            #Cv1 d_wc1
            conv1 = conv_layer(conv0, [self.KERNEL_SIZE, self.KERNEL_SIZE, 64, 128], 1, str(name +'d_wc1'))
            print('Cv1')
            print(sess.run(tf.shape(conv1)))
            #Cv2
            conv2 = conv_layer(conv1, [self.KERNEL_SIZE, self.KERNEL_SIZE, 128, 256], 1, str(name +'d_wc2'))
            print('Cv2')
            print(sess.run(tf.shape(conv2)))
            #Cv3
            conv3 = conv_layer(conv2, [self.KERNEL_SIZE, self.KERNEL_SIZE, 256, 512], 1, str(name +'d_wc3'))
            print('Cv3')
            print(sess.run(tf.shape(conv3)))
            #Cv4
            conv4 = conv_layer(conv3, [self.KERNEL_SIZE, self.KERNEL_SIZE, 512, 1], 1, str(name +'d_wc4'),False,False)
            print('Cv4')
            print(sess.run(tf.shape(conv4)))
            conv4=nn.sigmoid(conv4)
            return conv4
        
    def generator_2(self,x):
        print('making G2-network')
        sess=tf.Session()
        with tf.variable_scope('G2'):
            #Cv0
            conv0 =conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 4, 64], 1, 'g_wc0',True,False)
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
            conv11 =deconv2d(conv11, [self.BATCH,self.INPUT_SIZE, self.INPUT_SIZE, 3], self.KERNEL_SIZE, self.KERNEL_SIZE, 1, 1,"deConv11")    
            print('Cv11')
            print(sess.run(tf.shape(conv11)))
            conv11=nn.tanh(conv11)

            return conv11  

    def discriminator_2(self,x,name):
        print('making D2-network')
        sess=tf.Session()
        with tf.variable_scope('D2'):
        #Cv0
            conv0 = conv_layer(x, [self.KERNEL_SIZE, self.KERNEL_SIZE, 7, 64], 1, str(name +'d_wc0'),True,False)
            print('Cv1')
            print(sess.run(tf.shape(conv0)))
            #Cv1 d_wc1
            conv1 = conv_layer(conv0, [self.KERNEL_SIZE, self.KERNEL_SIZE, 64, 128], 1, str(name +'d_wc1'))
            print('Cv1')
            print(sess.run(tf.shape(conv1)))
            #Cv2
            conv2 = conv_layer(conv1, [self.KERNEL_SIZE, self.KERNEL_SIZE, 128, 256], 1, str(name +'d_wc2'))
            print('Cv2')
            print(sess.run(tf.shape(conv2)))
            #Cv3
            conv3 = conv_layer(conv2, [self.KERNEL_SIZE, self.KERNEL_SIZE, 256, 512], 1, str(name +'d_wc3'))
            print('Cv3')
            print(sess.run(tf.shape(conv3)))
            #Cv4
            conv4 = conv_layer(conv3, [self.KERNEL_SIZE, self.KERNEL_SIZE, 512, 1], 1, str(name +'d_wc4'),False,False)
            print('Cv4')
            print(sess.run(tf.shape(conv4)))
            conv4=nn.sigmoid(conv4)
            return conv4