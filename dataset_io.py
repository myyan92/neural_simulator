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

def data_parser(record, augment=False):
    features = tf.parse_single_example(
      record,
      features={
        'start': tf.FixedLenFeature([128], tf.float32),
        'action': tf.FixedLenFeature([128], tf.float32),
        'result': tf.FixedLenFeature([128], tf.float32),
        })

    start = tf.reshape(features['start'], tf.constant([64,2]))
    action = tf.reshape(features['action'], tf.constant([64,2]))
    result = tf.reshape(features['result'], tf.constant([64,2]))
    if augment:
        theta = tf.random.uniform([], -np.pi, np.pi)
        rotate = tf.stack([tf.cos(theta), tf.sin(theta), -tf.sin(theta), tf.cos(theta)])
        rotate = tf.reshape(rotate, (2,2))
        start = tf.matmul(start, rotate)
        action = tf.matmul(action, rotate)
        result = tf.matmul(result, rotate)
    return start, action, result


