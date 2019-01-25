# from PIL import Image
import os
import glob
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
        'start': tf.FixedLenFeature([128], tf.float32),
        'action': tf.FixedLenFeature([128], tf.float32),
        'result': tf.FixedLenFeature([128], tf.float32),
        })

    start = tf.reshape(features['start'], tf.constant([64,2]))
    action = tf.reshape(features['action'], tf.constant([64,2]))
    result = tf.reshape(features['result'], tf.constant([64,2]))
    return start, action, result


if __name__ == "__main__":

    tfrecords_filename = 'datasets/neuralsim_test_50.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    action_pattern = "/scr1/ylzhu/neural_simulator/data/%08d_act.txt"
    position_pattern = "/scr1/ylzhu/neural_simulator/data/%08d_%03d.txt"
    # num_curve = 10000
    # num_action = 10
    idx = 9000
    count = 0
    total_len = 0
    while idx < 9050:
        if not os.path.isfile(action_pattern % (idx)):
            print(action_pattern % (idx),"action not found")
            idx+=1
            continue
        statefile_len = len(glob.glob("/".join(position_pattern.split("/")[:-1])+"/{:08d}_[0-9][0-9][0-9]*.txt".format(idx)))
        total_len+=statefile_len
        print(idx,statefile_len,total_len,total_len//(count+1))
        for a in range(1,statefile_len):
            result = np.loadtxt(position_pattern % (idx,a))
            start = np.loadtxt(position_pattern % (idx,a-1))
            with open(action_pattern % (idx)) as f:
                line = f.readline()
            tokens = line.strip().split()
            action_node = int(tokens[0])
            action_x, action_y = float(tokens[1]), float(tokens[2])
            action = np.zeros_like(start)
            action[action_node,:]=np.array([action_x, action_y])
            record = data_writer(start, action, result)
            writer.write(record.SerializeToString())
        idx+=1
        count+=1

    writer.close()
    print('total:',total_len)
