from PIL import Image
import numpy as np
import tensorflow as tf
import pdb

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

def data_parser(record):
    features = tf.parse_single_example(
      record,
      features={
        'start': tf.FixedLenFeature([256], tf.float32),
        'action': tf.FixedLenFeature([256], tf.float32),
        'result': tf.FixedLenFeature([256], tf.float32),
        })

    start = tf.reshape(features['start'], tf.constant([128,2]))
    action = tf.reshape(features['action'], tf.constant([128,2]))
    result = tf.reshape(features['result'], tf.constant([128,2]))
    return start, action, result


if __name__ == "__main__":

    tfrecords_filename = 'datasets/neuralsim_test_s1ka10_b1.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    action_pattern = "/scr1/mengyuan/data/rollout_data_bending1/%04d_act.txt"
    position_pattern = "/scr1/mengyuan/data/rollout_data_bending1/%04d_%d.txt"
    num_curve = 10000
    num_action = 10
    for i in range(9000, 10000):
        for a in range(11):
            result = np.loadtxt(position_pattern % (i,a))
            start = np.loadtxt(position_pattern % (i,1))
            with open(action_pattern % (i)) as f:
                line = f.readline()
            tokens = line.strip().split()
            action_node = int(tokens[0])-1
            action_x, action_y = float(tokens[1]), float(tokens[2])
            action = np.zeros_like(start)
            action[action_node,:]=np.array([action_x, action_y])
            action *= (a-1)
            record = data_writer(start, action, result)
            writer.write(record.SerializeToString())

    writer.close()
