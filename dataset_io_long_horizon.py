import numpy as np
import tensorflow as tf
import pdb

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floatList_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def data_writer(s1,act,s2, gio, giu):
    gio = [g for g in gio if len(g)>0]
    giu = [g for g in giu if len(g)>0]
    if len(gio) > 0:
        gio = np.concatenate(gio, axis=0)
        giu = np.concatenate(giu, axis=0)
        length = gio.shape[0]
    else:
        gio = np.array([])
        giu = np.array([])
        length = 0
    gio = gio.astype(np.uint8).tostring()
    giu = giu.astype(np.uint8).tostring()
    record = tf.train.Example(features=tf.train.Features(feature={
        'start': _floatList_feature(s1.flatten().tolist()),
        'action': _floatList_feature(act.flatten().tolist()),
        'result': _floatList_feature(s2.flatten().tolist()),
        'index_over': _bytes_feature(gio),
        'index_under': _bytes_feature(giu),
        'length': _int64_feature(length),
    }))
    return record

def data_parser(record, dim, step=10, augment=True):
    features = tf.parse_single_example(
      record,
      features={
        'start': tf.FixedLenFeature([65*step*dim], tf.float32),
        'action': tf.FixedLenFeature([65*step*dim], tf.float32),
        'result': tf.FixedLenFeature([65*step*dim], tf.float32),
        'index_over': tf.FixedLenFeature([], tf.string),
        'index_under': tf.FixedLenFeature([], tf.string),
        'length': tf.FixedLenFeature([], tf.int64),
        })

    start = tf.reshape(features['start'], tf.constant([step,65,dim]))
    action = tf.reshape(features['action'], tf.constant([step,65,dim]))
    result = tf.reshape(features['result'], tf.constant([step,65,dim]))
    if augment:
        theta = tf.random_uniform([], -np.pi, np.pi)
        if dim == 2:
            rotate = tf.stack([tf.cos(theta), tf.sin(theta),
                              -tf.sin(theta), tf.cos(theta)])
            rotate = tf.reshape(rotate, (2,2))
        else dim==3:
            rotate = tf.stack([tf.cos(theta), tf.sin(theta), 0,
                              -tf.sin(theta), tf.cos(theta), 0,
                               0, 0, 1])
            rotate = tf.reshape(rotate, (3,3))
        start = tf.tensordot(start, rotate, 1)
        action = tf.tensordot(action, rotate, 1)
        result = tf.tensordot(result, rotate, 1)

    length = features['length']
    gio = tf.io.decode_raw(features['index_over'], tf.uint8)
    gio = tf.reshape(tf.cast(gio, tf.int32), [length,3])
    giu = tf.io.decode_raw(features['index_under'], tf.uint8)
    giu = tf.reshape(tf.cast(giu, tf.int32), [length,3])

    return start, action, result, gio, giu


def batch_map_fn(b_start, b_action, b_result, b_gio, b_giu):
    batch_index = tf.range(tf.shape(b_start)[0], dtype=tf.int32)
    batch_index = tf.reshape(batch_index, [-1,1,1])
    batch_index = tf.tile(batch_index, [1, tf.shape(b_gio)[1], 1])
    b_gio = tf.concat([batch_index, b_gio], axis=-1)
    b_giu = tf.concat([batch_index, b_giu], axis=-1)
    return b_start, b_action, b_result, b_gio, b_giu
