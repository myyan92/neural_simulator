# from PIL import Image
import os, random
import glob
import numpy as np
import tensorflow as tf
import pdb
from dataset_io import data_writer

old_data = glob.glob('/scr1/mengyuan/data/sim3d_sequence/*')
old_data = [d for d in old_data if os.path.isdir(d)]
new_data = glob.glob('/scr1/mengyuan/data/sim3d_sequence_sup/*')
new_data = [d for d in new_data if os.path.isdir(d)]

action_pattern = "%04d_act.txt"
position_pattern = "%04d_%03d.txt"

def add_data(data_dir, writer):
    idx = int(data_dir.split('/')[-1])
    act = np.loadtxt(os.path.join(data_dir, action_pattern % (idx)))
    for i,a in enumerate(act):
        if i ==0:
            continue
        result = np.loadtxt(os.path.join(data_dir, position_pattern % (idx,i+1)))
        start = np.loadtxt(os.path.join(data_dir, position_pattern % (idx,i)))
        assert(len(a)==3)
        if np.isnan(a).any():
            print(idx,i,a)
            continue
        action_node = int(a[0])
        if data_dir in old_data:
            move = np.array([[float(a[1]), -float(a[2])]])
        else:
            move = np.array([[float(a[1]), float(a[2])]])
        action = np.zeros_like(start)
        action[action_node,:] = move
        record = data_writer(start, action, result)
        writer.write(record.SerializeToString())

    return act.shape[0]-1


if __name__ == "__main__":

    data = old_data+new_data
    random.shuffle(data)
    train_data = [d for d in data if d[-1]!='0']
    test_data = [d for d in data if d[-1]=='0']
    print(len(train_data), len(test_data))
    pdb.set_trace()
    tfrecords_filename = 'datasets/neuralsim_train_simseq3d_new.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    total_len = 0
    for data in train_data:
        data_len = add_data(data, writer)
        total_len += data_len
    writer.close()
    print('train total:',total_len)

    tfrecords_filename = 'datasets/neuralsim_test_simseq3d_new.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    total_len = 0
    for data in test_data:
        data_len = add_data(data, writer)
        total_len += data_len
    writer.close()
    print('test total:',total_len)

