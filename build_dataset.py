# from PIL import Image
import os, random
import glob
import numpy as np
import tensorflow as tf
import pdb
from neural_simulator.dataset_io import data_writer


action_pattern = "%04d_act.txt"
position_pattern = "%04d_%03d.txt"

def add_data(data_dir, writer):
    idx = int(data_dir.split('/')[-1])
    act = np.loadtxt(os.path.join(data_dir, action_pattern % (idx)))
    for i,a in enumerate(act):
        result = np.loadtxt(os.path.join(data_dir, position_pattern % (idx,i+1)))
        start = np.loadtxt(os.path.join(data_dir, position_pattern % (idx,i)))
        if np.isnan(a).any():
            print(idx,i,a)
            continue
        action_node = int(a[0])
        action = np.zeros_like(start)
        action[action_node,:] = a[1:]
        record = data_writer(start, action, result)
        writer.write(record.SerializeToString())
    return act.shape[0]


if __name__ == "__main__":

    data = glob.glob('/scr1/mengyuan/data/data_simseq_2d/0*')
    random.shuffle(data)
    train_data = [d for d in data if d[-1]!='0']
    test_data = [d for d in data if d[-1]=='0']
    print(len(train_data), len(test_data))

    tfrecords_filename = 'datasets/neuralsim_train_simseq2d.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    total_len = 0
    for data in train_data:
        data_len = add_data(data, writer)
        total_len += data_len
    writer.close()
    print('train total:',total_len)

    tfrecords_filename = 'datasets/neuralsim_test_simseq2d.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    total_len = 0
    for data in test_data:
        data_len = add_data(data, writer)
        total_len += data_len
    writer.close()
    print('test total:',total_len)

