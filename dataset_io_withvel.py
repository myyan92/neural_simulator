import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def data_writer(s1,act,s2,vel):
    record = tf.train.Example(features=tf.train.Features(feature={
        'start': _floatList_feature(s1.flatten().tolist()),
        'action': _floatList_feature(act.flatten().tolist()),
        'result': _floatList_feature(s2.flatten().tolist()),
        'vel': _floatList_feature(vel.flatten().tolist())}))
    return record

def data_parser(record, augment=True):
    features = tf.parse_single_example(
      record,
      features={
        'start': tf.FixedLenFeature([128], tf.float32),
        'action': tf.FixedLenFeature([128], tf.float32),
        'result': tf.FixedLenFeature([128], tf.float32),
        'vel': tf.FixedLenFeature([65*64*2], tf.float32),
        })

    start = tf.reshape(features['start'], tf.constant([64,2]))
    action = tf.reshape(features['action'], tf.constant([64,2]))
    result = tf.reshape(features['result'], tf.constant([64,2]))
    vel = tf.reshape(features['vel'], tf.constant([65*64,2]))
    if augment:
        theta = tf.random_uniform([], -np.pi, np.pi)
        rotate = tf.stack([tf.cos(theta), tf.sin(theta), -tf.sin(theta), tf.cos(theta)])
        rotate = tf.reshape(rotate, (2,2))
        start = tf.matmul(start, rotate)
        action = tf.matmul(action, rotate)
        result = tf.matmul(result, rotate)
        vel = tf.matmul(vel, rotate)
    vel = tf.reshape(vel, tf.constant([65,64,2]))
    sum_vel = tf.cumsum(vel, axis=0)
    pos = start + sum_vel/64.0
    state = tf.concat([pos[1:],vel[:-1]], axis=-1)
    return start, action, result, state, vel[1:]


def data_parser_raw(record, augment=True):
    features = tf.parse_single_example(
      record,
      features={
        'pos': tf.FixedLenFeature([128], tf.float32),
        'vel': tf.FixedLenFeature([128], tf.float32),
        'pred_vel': tf.FixedLenFeature([128], tf.float32),
        })

    start_pos = tf.reshape(features['pos'], tf.constant([64,2]))
    start_vel = tf.reshape(features['vel'], tf.constant([64,2]))
    pred_vel = tf.reshape(features['pred_vel'], tf.constant([64,2]))
    if augment:
        theta = tf.random_uniform([], -np.pi, np.pi)
        rotate = tf.stack([tf.cos(theta), tf.sin(theta), -tf.sin(theta), tf.cos(theta)])
        rotate = tf.reshape(rotate, (2,2))
        start_pos = tf.matmul(start_pos, rotate)
        start_vel = tf.matmul(start_vel, rotate)
        pred_vel = tf.matmul(pred_vel, rotate)
    return start_pos, start_vel, pred_vel

