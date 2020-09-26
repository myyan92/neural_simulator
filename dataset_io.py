import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def data_writer(s1,act,s2):
    record = tf.train.Example(features=tf.train.Features(feature={
        'start': _floatList_feature(s1.flatten().tolist()),
        'action': _floatList_feature(act.flatten().tolist()),
        'result': _floatList_feature(s2.flatten().tolist())}))
    return record

def data_parser(record, dim, augment=True):
    features = tf.parse_single_example(
      record,
      features={
        'start': tf.FixedLenFeature([65*dim], tf.float32),
        'action': tf.FixedLenFeature([65*dim], tf.float32),
        'result': tf.FixedLenFeature([65*dim], tf.float32),
        })

    start = tf.reshape(features['start'], tf.constant([65,dim]))
    action = tf.reshape(features['action'], tf.constant([65,dim]))
    result = tf.reshape(features['result'], tf.constant([65,dim]))
    if augment:
        theta = tf.random_uniform([], -np.pi, np.pi)
        if dim==2:
            rotate = tf.stack([tf.cos(theta), tf.sin(theta),
                              -tf.sin(theta), tf.cos(theta)])
            rotate = tf.reshape(rotate, (2,2))
        if dim==3:
            rotate = tf.stack([tf.cos(theta), tf.sin(theta), 0,
                              -tf.sin(theta), tf.cos(theta), 0,
                               0, 0, 1])
            rotate = tf.reshape(rotate, (3,3))

        start = tf.matmul(start, rotate)
        action = tf.matmul(action, rotate)
        result = tf.matmul(result, rotate)
    return start, action, result


