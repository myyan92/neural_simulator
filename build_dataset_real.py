# from PIL import Image
import os
import glob
import numpy as np
import tensorflow as tf
from neural_simulator.dataset_io import data_writer
import matplotlib.pyplot as plt
import pdb


if __name__ == "__main__":

#    tfrecords_filename = 'datasets/neuralsim_train_real_with_occlusions.tfrecords'
    tfrecords_filename = 'datasets/neuralsim_test_real_no_occlusions.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    total_len = 0
    for run in range(1,10):
        #data = np.load('/scr1/mengyuan/data/real_rope_with_occlusion-new/run_%d/history_perception.npz'%(run))
        data = np.load('/scr1/mengyuan/data/real_rope_ours_2/seq_m%d_2/history_perception.npz'%(run))
        states = data['perception']
        actions = data['actions']
        for i,a in enumerate(actions):
            #if np.linalg.norm(a[2:4]) < 1e-4:
            #    continue # pick or place action
            result = states[i+1].copy()
            start = states[i].copy()
            dists = np.linalg.norm(start-a[0:2], axis=1)
            action_node = np.argmin(dists)
            #action_move = actions[i+1][0:2]-a[0:2]
            action_move = a[2:4]
            action = np.zeros_like(start)
            action[action_node,:]=action_move
            start[:,0] -= 0.5 # match simulation data coordinate range
            result[:,0] -= 0.5 
#            plt.plot(start[:,0], start[:,1])
#            plt.plot(result[:,0], result[:,1])
#            plt.plot([start[action_node,0], start[action_node,0]+action_move[0]],
#                     [start[action_node,1], start[action_node,1]+action_move[1]])
#            plt.axis([-0.5,0.5,-0.5,0.5])
#            plt.show()
            record = data_writer(start, action, result)
            writer.write(record.SerializeToString())
            total_len += 1
        print(run,total_len)

    writer.close()
    print('total:',total_len)
