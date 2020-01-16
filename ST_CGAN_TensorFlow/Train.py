from Image_loader import *
from Model import *
import tensorflow as tf
from tensorflow import keras
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Utils import *

IMAGE_SIZE = 256 
BATCH_SIZE = 1
#EPOCH = 50000000
EPOCH = 10000

lambda1 = 5
lambda2 = 0.1
lambda3 = 0.1
#INIT_G_LEARNING_RATE = 1e-5 #1E-5
#INIT_LEARNING_RATE = 1e-4 #1E-4
#MOMENTUM = 0.9
#LEARNING_RATE=0.001
LEARNING_RATE = 0.0002
BETA1 = 0.5
log_path = './graph/logs' # path to tensorboard graph
TFRECORD_NAME = 'ISTD_Dataset_256x256.tfrecord'
criterion1 = tf.keras.losses.BinaryCrossentropy()
#criterion2 = torch.nn.L1Loss()


def train(backupFlag):
    # setting random seed and reset graphs

    G_LEARNING_RATE = 1e-5 #1e-5
    LEARNING_RATE = 1e-4 #1e-4

    #tf.set_random_seed(1111)
    #tf.reset_default_graph()

    ##calc step number
    step_num = int(6 / BATCH_SIZE)

    # place holders for shadow image input and shadow free image input
    with tf.variable_scope('Data_Input'):
        shadow = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3], name ='Shadow_image')
        shadow_mask = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1], name = 'Shadow_mask_image')
        shadow_free = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3], name = 'Shadow_free_image')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        lr_g = tf.placeholder(tf.float32, name='learn_rate_g')
        lr = tf.placeholder(tf.float32, name ='learn_rate_as')
    # init network model
    model = ST_CGAN(shadow,shadow_mask)
    #
    #with tf.Session() as sess:

    #    ## ==================== for tensorflow debugger ========================= ##
    #    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    #    ## ====================================================================== ##
    #    epoch = tf.Variable(0, name='epoch', trainable=False) # epoch     
    #    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.G1_loss)
    #    # loading dataset ...
    #    reader = tf.TFRecordReader()
    #    shadow_input,shadow_mask_input,shadow_free_input = read_tfrecord_triple_img(reader, TFRECORD_NAME, False)


    # loading dataset ...
    reader = tf.TFRecordReader()
    shadow_input,shadow_mask_input,shadow_free_input = read_tfrecord_triple_img(reader, TFRECORD_NAME, False)

    # shuffle batch ...
    shadow_batch,shadow_mask_batch,shadow_free_batch = tf.train.shuffle_batch([shadow_input,shadow_mask_input,shadow_free_input],
                                           BATCH_SIZE,
                                           capacity=10*BATCH_SIZE,
                                           min_after_dequeue = 2*BATCH_SIZE,
                                           num_threads=2,
                                           enqueue_many=False )   
    
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.G1_loss)
    saver = tf.compat.v1.train.Saver()
    dataset_length=sum(1 for _ in tf.python_io.tf_record_iterator(TFRECORD_NAME))

    init = tf.global_variables_initializer()
    #local= tf.local_variables_initializer()

    #threads coordinator
    coord = tf.train.Coordinator()
   # threads = tf.train.start_queue_runners(coord = coord)
    #tf.train.start_queue_runners()

    with tf.Session() as session:
        threads = tf.train.start_queue_runners(sess=session,coord=coord)
        session.run(init)
        for epoch in range(EPOCH):
                    #for i in range(cfg.ITERATIONS):
                    for i in tqdm.tqdm(range(dataset_length)):
                       # load image batched
                        img_shadow_seq, img_mask_seq, img_gt_seq = session.run([shadow_batch,shadow_mask_batch,shadow_free_batch])
                        #img_mask_seq=tf.convert_to_tensor(tf.image.rgb_to_grayscale(img_mask_seq))
                        #img_mask_seq = img_mask_seq.convert('LA')

                        _,loss = session.run([optimizer,model.G1_loss],
                                            feed_dict={
                                            shadow: img_shadow_seq,
                                            shadow_mask: img_mask_seq,
                                            shadow_free:  img_gt_seq,
                                            keep_prob: 0.5,
                                            lr: LEARNING_RATE,
                                            lr_g: G_LEARNING_RATE
                                            
                                            })
                        print('loss in one epoch: {}'.format(loss))

                        g1_output = session.run([model.G1],
                                    feed_dict={
                                    shadow: img_shadow_seq,
                                    shadow_mask: img_mask_seq,
                                    shadow_free:  img_gt_seq,
                                    keep_prob: 1.0,
                                    lr: LEARNING_RATE,
                                    lr_g:G_LEARNING_RATE
                                    })
                        ###tf.clip_by_value(g1_output[0], 0,255)     
                        #out = g1_output[0].astype(np.uint8)
                        #img = Image.fromarray(out)
                        #img.show()
                        #img_frombytes(out)
                        m = np.squeeze(g1_output) # you can give axis attribute if you wanna squeeze in specific dimension
                        pilimg = Image.fromarray(m)
                        pilimg.show()

                       #session.run(optimizer, 
                        #                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: True, hold_prob:cfg.TRAIN_HOLD_PROBABILITY})
                        #print("iterate:", (i))


                    #image_batch, labels_batch = ld.get_train_data(cfg.BATCH_SIZE)
                    ## image_batch = (image_batch - 128) / 128
                    #test_acc = session.run(accuracy, 
                    #                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: False, hold_prob:cfg.TEST_HOLD_PROBABILITY})
                    coord.request_stop()
                    coord.join(threads)
                    session.close()
                    print("Epoch:", (epoch))
                    
     




train(False)