from PIL import Image
import numpy as np
import tensorflow as tf
from neural_simulator.dataset_io import data_writer
import pdb


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
