# from PIL import Image
import os, random
import glob
import numpy as np
import tensorflow as tf
import pdb
from dataset_io import data_writer
from physbam_python.utils import hasCollision

#data = glob.glob('/scr1/mengyuan/data/simseq3d_with_intersections_has_friction/*')
data = glob.glob('/scr1/mengyuan/data/simseq3d_topochange/*')
data = [d for d in data if os.path.isdir(d)]

action_pattern = "%04d_act.txt"
position_pattern = "%04d_%03d.txt"

def add_data(data_dir, writer_nointersect, writer_hasintersect, writer_mixed):
    idx = int(data_dir.split('/')[-1])
    act = np.loadtxt(os.path.join(data_dir, action_pattern % (idx)))
    for i,a in enumerate(act):
        result = np.loadtxt(os.path.join(data_dir, position_pattern % (idx,i+1)))
        start = np.loadtxt(os.path.join(data_dir, position_pattern % (idx,i)))
        #assert(len(a)==3)
        if np.isnan(a).any():
            print(idx,i,a)
            continue
        action_node = int(a[0])
        move = a[1:]
#        if np.linalg.norm(move[:2])<1e-4:
#            continue
        # adding end-effector rotation info.
        tangent_start = start[min(action_node+1,start.shape[0]-1)] - start[max(action_node-1,0)]
        if np.linalg.norm(tangent_start) < 1e-4:
            pdb.set_trace()
        tangent_start = tangent_start / np.linalg.norm(tangent_start)
        tangent_result = result[min(action_node+1,start.shape[0]-1)] - result[max(action_node-1,0)]
        if np.linalg.norm(tangent_result) < 1e-4:
            pdb.set_trace()
        tangent_result = tangent_result / np.linalg.norm(tangent_result)
        if action_node < 4 or action_node > 60:
            norm_move = np.array([move[0], move[1], 0.0])
            norm_move = norm_move / np.linalg.norm(norm_move)
            tangent_result[2]=0
            tangent_result = tangent_result / np.linalg.norm(tangent_result)
            if action_node < 4 and np.sum(norm_move*tangent_result)>-0.9:
#                print(idx,i)
                pass
            if action_node > 60 and np.sum(norm_move*tangent_result)<0.9:
#                print(idx,i)
                pass
        action_angle = np.arctan2(tangent_result[0], tangent_result[1]) - \
                       np.arctan2(tangent_start[0], tangent_start[1])
        action_angle = np.array([np.sin(action_angle), np.cos(action_angle)])
        action = np.zeros((start.shape[0],start.shape[1]+2))
        action[action_node,:-2] = move
        action[action_node,-2:] = action_angle
        #action = np.zeros_like(start)
        #action[action_node,:] = move
        has_intersection = hasCollision(start)
        record = data_writer(start, action, result)
        if has_intersection:
            writer_hasintersect.write(record.SerializeToString())
        else:
            writer_nointersect.write(record.SerializeToString())
        writer_mixed.write(record.SerializeToString())
    return act.shape[0]-1


if __name__ == "__main__":

    train_data = [d for d in data if d[-1]!='0']
    test_data = [d for d in data if d[-1]=='0']
    print(len(train_data), len(test_data))
    pdb.set_trace()
#    tfrecords_filename = 'datasets/neuralsim_train_simseq3d_new2_nointersection_friction.tfrecords'
    tfrecords_filename = 'datasets/neuralsim_train_simseq3d_topochange_tmp1.tfrecords'
    writer1 = tf.python_io.TFRecordWriter(tfrecords_filename)
#    tfrecords_filename = 'datasets/neuralsim_train_simseq3d_new2_hasintersection_friction.tfrecords'
    tfrecords_filename = 'datasets/neuralsim_train_simseq3d_topochange_tmp2.tfrecords'
    writer2 = tf.python_io.TFRecordWriter(tfrecords_filename)
#    tfrecords_filename = 'datasets/neuralsim_train_simseq3d_new2_mixed_friction.tfrecords'
    tfrecords_filename = 'datasets/neuralsim_train_simseq3d_topochange.tfrecords'
    writer3 = tf.python_io.TFRecordWriter(tfrecords_filename)
    total_len = 0
    for data in train_data:
        data_len = add_data(data, writer1, writer2, writer3)
        total_len += data_len
    writer1.close()
    writer2.close()
    writer3.close()
    print('train total:',total_len)

#    tfrecords_filename = 'datasets/neuralsim_test_simseq3d_new2_nointersection_friction.tfrecords'
    tfrecords_filename = 'datasets/neuralsim_test_simseq3d_topochage_tmp1.tfrecords'
    writer1 = tf.python_io.TFRecordWriter(tfrecords_filename)
#    tfrecords_filename = 'datasets/neuralsim_test_simseq3d_new2_hasintersection_friction.tfrecords'
    tfrecords_filename = 'datasets/neuralsim_test_simseq3d_topochage_tmp2.tfrecords'
    writer2 = tf.python_io.TFRecordWriter(tfrecords_filename)
#    tfrecords_filename = 'datasets/neuralsim_test_simseq3d_new2_mixed_friction.tfrecords'
    tfrecords_filename = 'datasets/neuralsim_test_simseq3d_topochage.tfrecords'
    writer3 = tf.python_io.TFRecordWriter(tfrecords_filename)
    total_len = 0
    for data in test_data:
        data_len = add_data(data, writer1, writer2, writer3)
        total_len += data_len
    writer1.close()
    writer2.close()
    writer3.close()
    print('test total:',total_len)

