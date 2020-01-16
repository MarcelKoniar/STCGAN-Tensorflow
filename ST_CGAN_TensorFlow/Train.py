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
    model = ST_CGAN(shadow,shadow_mask,shadow_free)

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
    
    #optimizerD1 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.D1_loss*lambda2)
    #optimizerD2 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.D2_loss*lambda3)
    optimizerG1 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize (model.G1_loss) #maximize (descent gradient) https://stackoverflow.com/questions/38235648/is-there-an-easy-way-to-implement-a-optimizer-maximize-function-in-tensorflow
    optimizerG2 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize (model.G2_loss*lambda1)
    saver = tf.compat.v1.train.Saver()
    dataset_length=sum(1 for _ in tf.python_io.tf_record_iterator(TFRECORD_NAME))

    init = tf.global_variables_initializer()

    #threads coordinator
    coord = tf.train.Coordinator()
    ##testGPU
    #device_name = tf.test.gpu_device_name()
    #print(tf.VERSION)
    #if device_name != '/device:GPU:0':
    #  raise SystemError('GPU device not found')
    #print('Found GPU at: {}'.format(device_name))

    with tf.Session() as session:
        print('check if there is backup')
        if backupFlag and tf.train.get_checkpoint_state('./backup'):
            print('backup file loading ...')
            saver = tf.compat.v1.train.Saver()
            saver.restore(session, './backup/latest')
        else:
            print('no backup ...')
        # restore check-point if it exits
        #saver = tf.compat.v1.train.Saver()
        #could_load, checkpoint_counter = load(saver,'./backup/latest',session)
        #if could_load:
        #    start_epoch = (int)(checkpoint_counter / self.num_batches)
        #    start_batch_id = checkpoint_counter - start_epoch * self.num_batches
        #    counter = checkpoint_counter
        #    print(" [*] Load SUCCESS")
        #else:
        #    start_epoch = 0
        #    start_batch_id = 0
        #    counter = 1
        #    print(" [!] Load failed...")

        threads = tf.train.start_queue_runners(sess=session,coord=coord)
        session.run(init)
        for epoch in range(EPOCH):
                    #for i in range(cfg.ITERATIONS):
                    for i in tqdm.tqdm(range(dataset_length)):
                       # load image batched
                        img_shadow_seq, img_mask_seq, img_gt_seq = session.run([shadow_batch,shadow_mask_batch,shadow_free_batch])
                      

                        #_,lossD1 = session.run([optimizerD1,model.D1_loss],
                        #                    feed_dict={
                        #                    shadow: img_shadow_seq,
                        #                    shadow_mask: img_mask_seq,
                        #                    shadow_free:  img_gt_seq,
                        #                    keep_prob: 0.5,
                        #                    lr: LEARNING_RATE,
                        #                    lr_g: G_LEARNING_RATE
                                            
                        #                   })
                        #_,lossD2 = session.run([optimizerD2,model.D2_loss],
                        #                    feed_dict={
                        #                    shadow: img_shadow_seq,
                        #                    shadow_mask: img_mask_seq,
                        #                    shadow_free:  img_gt_seq,
                        #                    keep_prob: 0.5,
                        #                    lr: LEARNING_RATE,
                        #                    lr_g: G_LEARNING_RATE
                                            
                        #                   })
                        _,lossG1,g1_output = session.run([optimizerG1,model.G1_loss,model.G1],
                                            feed_dict={
                                            shadow: img_shadow_seq,
                                            shadow_mask: img_mask_seq,
                                            shadow_free:  img_gt_seq,
                                            keep_prob: 0.5,
                                            lr: LEARNING_RATE,
                                            lr_g: G_LEARNING_RATE
                                            
                                            })
                        _,lossG2,g2_output = session.run([optimizerG2,model.G2_loss,model.G2],
                                            feed_dict={
                                            shadow: img_shadow_seq,
                                            shadow_mask: img_mask_seq,
                                            shadow_free:  img_gt_seq,
                                            keep_prob: 0.5,
                                            lr: LEARNING_RATE,
                                            lr_g: G_LEARNING_RATE
                                            
                                            })
                        #print('loss in {} iteration: {}'.format(i,loss))

                        #print('D1 loss: {}   G1 loss: {}    D2 loss: {}   G2 loss: {}    in iteration: {}, epoch: {}'.format(lossD1,lossG1,lossD2,lossG2,i,epoch))
                        #print('D1 loss: {}   G1 loss: {}    in iteration: {}, epoch: {}'.format(0,lossG1,i,epoch))
                        #print('D1 loss: {}   G2 loss: {}    in iteration: {}, epoch: {}'.format(0,lossG2,i,epoch))
                        print('D1 loss: {}   G1 loss: {}    D2 loss: {}   G2 loss: {}    in iteration: {}, epoch: {}'.format(0,lossG1,0,lossG2,i,epoch))




                        iminput=decode_image(g1_output)
                        cv2.imwrite('result/train G1/it{}_epoch{}.png'.format(i,epoch),iminput)
                        iminput=decode_image(g2_output)
                        cv2.imwrite('result/train G2/it{}_epoch{}.png'.format(i,epoch),iminput)
                    # saving data
                    saver = tf.compat.v1.train.Saver()
                    saver.save(session, './backup/latest', write_meta_graph = False)
                    if epoch == EPOCH:
                       saver.save(session,'./backup/fully_trained',write_meta_graph = False)
                     

                    
     




train(False)