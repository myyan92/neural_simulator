import os, random
import glob
import numpy as np
import tensorflow as tf
import pdb
from dataset_io_long_horizon import data_writer
from topology.state_2_topology import find_intersections

action_pattern = "%04d_act.txt"
position_pattern = "%04d_%03d.txt"

def add_data(horizon, data_dir, writer):
    idx = int(data_dir.split('/')[-1])
    act = np.loadtxt(os.path.join(data_dir, action_pattern % (idx)))
    states = [np.loadtxt(os.path.join(data_dir, position_pattern % (idx,i))) for i in range(len(act))]
    states = np.array(states)

    intersections = [find_intersections(state) for state in states]
    gather_index_over = [[ [it[0],2] if it[2]==1 else [it[1],2]
                           for it in intersect]
                           for t,intersect in enumerate(intersections)]
    gather_index_under = [[ [it[1],2] if it[2]==1 else [it[0],2]
                            for it in intersect]
                            for t,intersect in enumerate(intersections)]

    for i, index in enumerate(gather_index_over):
        if len(index)==0:
            continue
        index = np.array(index)[:,:,np.newaxis]
        index = np.tile(index, (1,1,4))
        index[:,0,:] += np.array([-1,0,1,2])
        index = np.clip(index, 0, 63)
        index = np.tile(index[:,:,:,np.newaxis], (1,1,1,4))
        index = index.transpose((0,2,3,1)).reshape((-1,2))
        gather_index_over[i] = index
    for i, index in enumerate(gather_index_under):
        if len(index)==0:
            continue
        index = np.array(index)[:,:,np.newaxis]
        index = np.tile(index, (1,1,4))
        index[:,0,:] += np.array([-1,0,1,2])
        index = np.clip(index, 0, 63)
        index = np.tile(index[:,:,:,np.newaxis], (1,1,1,4))
        index = index.transpose((0,2,3,1)).reshape((-1,2))
        gather_index_under[i] = index

    actions = np.zeros_like(states)
    for i,a in enumerate(act):
        action_node = int(a[0])
        move = a[1:]
        actions[i,action_node] = move

    for i in range(0, len(act)-horizon, horizon//2):
        start = states[i:i+horizon]
        action = actions[i:i+horizon]
        result = states[i+1:i+1+horizon]
        gio = gather_index_over[i+1:i+1+horizon]
        gio = [np.insert(g, 0, t, axis=1) if len(g)>0 else [] for t,g in enumerate(gio)]
        giu = gather_index_under[i+1:i+1+horizon]
        giu = [np.insert(g, 0, t, axis=1) if len(g)>0 else [] for t,g in enumerate(giu)]
        record = data_writer(start, action, result, gio, giu)
        writer.write(record.SerializeToString())
    return act.shape[0]-1


if __name__ == "__main__":
  topo_total = ['data_0to1_R1_left-1_sign-1', 'data_0to1_R2_left-1_over_before_under-1',
                'data_1to2_R2_left-1_diff', 'data_1to2_R2_left-1_over_before_under-1', 'data_1to2_cross_endpoint-over_sign-1']
  horizon = 10
  train_data = []
  test_data = []
  for topo in topo_total:
    print(topo)
    data = glob.glob('/home/genli/gen_data_long_horizon/%s/*'%topo)
    data = [d for d in data if os.path.isdir(d)]
    train_data_ = [d for d in data if d[-1]!='0']
    test_data_ = [d for d in data if d[-1]=='0']
    print(len(train_data_), len(test_data_))
    train_data.extend(train_data_)
    test_data.extend(test_data_)
  print(len(train_data), len(test_data))

  from random import shuffle
  shuffle(train_data)
  shuffle(test_data)
  tfrecords_filename = 'datasets/neuralsim_train_simseq3d_long_horizon_reg.tfrecords'
  writer = tf.python_io.TFRecordWriter(tfrecords_filename)
  total_len = 0
  for data in train_data:
      data_len = add_data(horizon, data, writer)
      total_len += data_len
  writer.close()
  print('train total:',total_len)

  tfrecords_filename = 'datasets/neuralsim_test_simseq3d_long_horizon_reg.tfrecords'
  writer = tf.python_io.TFRecordWriter(tfrecords_filename)
  total_len = 0
  for data in test_data:
      data_len = add_data(horizon, data, writer)
      total_len += data_len
  writer.close()
  print('test total:',total_len)

