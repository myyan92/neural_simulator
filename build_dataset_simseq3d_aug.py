# from PIL import Image
import os
from functools import partial
import glob
import numpy as np
import tensorflow as tf
import pdb


def rotate2d(v,theta):
    if np.isclose(theta,np.pi/2):
        return np.array([-v[1], v[0]])
    if np.isclose(theta,-np.pi/2):
        return np.array([v[1], -v[0]])
    return np.array([v[0]*np.cos(theta)-v[1]*np.sin(theta),
                     v[0]*np.sin(theta)+v[1]*np.cos(theta)])

def aug_data(data, flipx=False, flipy=False, theta=0.0):
    assert(data.ndim==2 and data.shape[1]==2)
    if flipx:
        data[:,0] = -data[:,0]
    if flipy:
        data[:,1] = -data[:,1]
    fp = partial(rotate2d, theta=theta)
    np.apply_along_axis(fp, 1, data) 
    #  for i, d in enumerate(data):
    #      data[i,:] = rotate2d(d,theta)
    return data

if __name__ == "__main__":
    for it in range(20):
        idx = 9000+it*10
        count = 0
        total_len = 0
        tfrecords_filename = 'datasets/neuralsim_test_aug_{}.tfrecords'.format(idx)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        action_pattern = "/scr1/mengyuan/data/sim3d_sequence/%04d/%04d_act.txt"
        position_pattern = "/scr1/mengyuan/data/sim3d_sequence/%04d/%04d_%03d.txt"
        # num_curve = 10000
        # num_action = 10
        while count < 1: #idx < 10000:
            if not os.path.isfile(action_pattern % (idx,idx)):
                print(action_pattern % (idx,idx),"action not found")
                idx+=1
                continue
            flipx = True if np.random.randint(2) else False
            flipy = True if np.random.randint(2) else False
            rotate = np.random.uniform(-np.pi,np.pi)
            act = np.loadtxt(action_pattern % (idx,idx))
            statefile_len = act.shape[0]
            total_len+=statefile_len
            print(flipx,flipy,rotate)
            print(it,idx,statefile_len,total_len,total_len//(count+1))
            for i,a in enumerate(act):
                if i ==0:
                    continue
                result = np.loadtxt(position_pattern % (idx,idx,i+1))
                start = np.loadtxt(position_pattern % (idx,idx,i))
                result = aug_data(result, flipx=flipx, flipy=flipy, theta=rotate)
                start = aug_data(start, flipx=flipx, flipy=flipy, theta=rotate)
                assert(len(a)==3)
                if np.isnan(a).any():
                    print(idx,i,a)
                    continue
                action_node = int(a[0])
                action_x, action_y = aug_data(np.array([[float(a[1]), -float(a[2])]]),
                                              flipx=flipx, flipy=flipy, theta=rotate).flatten()
                action = np.zeros_like(start)
                action[action_node,:]=np.array([action_x, action_y])
                record = data_writer(start, action, result)
                writer.write(record.SerializeToString())
            idx+=1
            count+=1

            writer.close()
            print('total:',total_len)
