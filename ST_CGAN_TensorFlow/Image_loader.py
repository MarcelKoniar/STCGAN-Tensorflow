import glob
import os
from PIL import Image
#import torchvision.transforms as transforms
#import torch.utils.data as DATA
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform 
import tensorflow.nn as nn

TFRECORD_NAME = 'ISTD_Dataset_256x256.tfrecord'
IMAGE_SIZE = 256
# Just disables the warning, doesn't enable AVX/FMA
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_path_list():
    dataset = []
    original_img_rpath = 'H:/Diplomka/ST_CGAN/ST_CGAN/ISTD_Dataset/train/train_A'
    shadow_mask_rpath = 'H:/Diplomka/ST_CGAN/ST_CGAN/ISTD_Dataset/train/train_B'
    shadow_free_img_rpath = 'H:/Diplomka/ST_CGAN/ST_CGAN/ISTD_Dataset/train/train_C'
    for img_path in glob.glob(os.path.join(original_img_rpath, '*.png')):
        basename = os.path.basename(img_path)
        original_img_path = os.path.join(original_img_rpath, basename)
        shadow_mask_path = os.path.join(shadow_mask_rpath, basename)
        shadow_free_img_path = os.path.join(shadow_free_img_rpath, basename)
        #print(original_img_path, shadow_mask_path, shadow_free_img_path)
        dataset.append([original_img_path, shadow_mask_path, shadow_free_img_path])
    #print(dataset)
    return dataset

def make_tfrecord():
    # making a list of image
    image_list = make_path_list()
    # writing tf record
    print('tf record writing start')
    writer = tf.python_io.TFRecordWriter(TFRECORD_NAME)

    index = 0
    for img_triple in image_list:
        index += 1
        #print('number: {}'.format(index))
        write_to_tfrecord_triple_img(writer, img_triple[0],img_triple[1],img_triple[2],TFRECORD_NAME)

    writer.close()


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_tfrecord_triple_img(writer, binary_image1, binary_image2, binary_image3, tfrecord_file):
    # Create a feature
    binary_image1 = load_image(binary_image1)
    binary_image2 = load_image(binary_image2,True)
    binary_image3 = load_image(binary_image3)
    #binary_image1 = get_image_binary(binary_image1)
    #binary_image2 = get_image_binary(binary_image2)
    #binary_image3 = get_image_binary(binary_image3)

    example = tf.train.Example(features=tf.train.Features(feature={
                'shadow': _bytes_feature(tf.compat.as_bytes(binary_image1.tostring())),
                'shadow_mask': _bytes_feature(tf.compat.as_bytes(binary_image2.tostring())),
                'shadow_free': _bytes_feature(tf.compat.as_bytes(binary_image3.tostring()))
                }))
    writer.write(example.SerializeToString())


def load_image(addr, is_mask=False):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    if(is_mask==True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   # img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]

    return img

def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    # image = Image.open(filename)
    # image = np.asarray(image, np.uint8)
    # shape = np.array(image.shape, np.int32)

    image = cv2.imread(filename)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # cv.imshow('test',image)
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # cv.waitKey()
    image = np.asarray(image, np.uint8)
    #shape = np.array(image.shape, np.int32)
    #return shape.tobytes(), image.tostring() #image.tobytes() # convert image to raw data bytes in the array.
    return image

#make_tfrecord()
def read_from_tfrecord_triple_img(reader, filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'shadow': tf.FixedLenFeature([], tf.string),
                            'shadow_mask': tf.FixedLenFeature([], tf.string),
                            'shadow_free': tf.FixedLenFeature([], tf.string),

                        }, name='features')

    # image was saved as uint8, so we have to decode as uint8.
    shadow = tf.decode_raw(tfrecord_features['shadow'], tf.uint8)
    shadow_mask    = tf.decode_raw(tfrecord_features['shadow_mask'],    tf.uint8)
    shadow_free    = tf.decode_raw(tfrecord_features['shadow_free'],    tf.uint8)
   

    shadow = tf.reshape(shadow, [IMAGE_SIZE, IMAGE_SIZE,3])
    shadow_mask = tf.reshape(shadow_mask, [IMAGE_SIZE, IMAGE_SIZE,1])#1
    shadow_free = tf.reshape(shadow_free, [IMAGE_SIZE, IMAGE_SIZE,3])
    return shadow, shadow_mask,shadow_free
#
def read_tfrecord_triple_img(reader, tfrecord_file,vis):
    shadow,shadow_mask,shadow_free = read_from_tfrecord_triple_img(reader, [tfrecord_file])
    if vis:
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #for i in range(4):
            for i in tf.python_io.tf_record_iterator(tfrecord_file):
                s,m,f = sess.run([shadow,shadow_mask,shadow_free])
                pilimg = Image.fromarray(s)
                pilimg.show()
                m = np.squeeze(m) # you can give axis attribute if you wanna squeeze in specific dimension
                pilimg = Image.fromarray(m)
                pilimg.show()
                pilimg = Image.fromarray(f)
                pilimg.show()
      
            coord.request_stop()
            coord.join(threads)
            sess.close()

    return shadow, shadow_mask,shadow_free

def read_dataset():
    reader = tf.TFRecordReader()
    read_tfrecord_triple_img(reader,TFRECORD_NAME,True)

#make_tfrecord()
#read_dataset()
