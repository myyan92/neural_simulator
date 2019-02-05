# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for visualizing the prediction model."""

import numpy as np
import tensorflow as tf
from PIL import Image
import argparse, gin
from sim_integration.hybrid_inference import HybridInference
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
from MPC_2d.dynamic_models import *
import matplotlib.pyplot as plt
import os, pdb, time, glob


model = Model_STNv2(vgg16_npy_path='/scr-ssd/mengyuan/TF_cloth2d/models/vgg16_weights.npz',
                    fc_sizes=[1024,1024,256],
                    save_dir='./tmp')
inferencer = HybridInference(model,
                             pred_target='node',
                             snapshot='/scr-ssd/mengyuan/TF_cloth2d/simseq_data_pred_node_STN_2_large_augmented/model-38',
                             memory=True, batch_size=50)

def load_sim_sequence(dir, use_vision=False):
    index = dir.split('/')[-1]
    with open(os.path.join(dir, index+'_act.txt')) as f:
        lines = f.readlines()
    actions = [l.strip().split() for l in lines]
    actions = [(int(l[0]), float(l[1]), float(l[2])) for l in actions]
    actions = [(ac[0], np.array([ac[1], ac[2]])) for ac in actions]
    num_actions = len(actions)
    states = []
    if use_vision:
        for i in range(num_actions+1):
            im = Image.open(os.path.join(dir, index+'_%03d.png'%(i)))
            image = np.array(im)[::-1,:,:]
            states.append(image)
    else:
        for i in range(num_actions+1):
            state = np.loadtxt(os.path.join(dir, index+'_%03d.txt'%(i)))
            state[:,1]=-state[:,1]
            states.append(state)

    return states[1:], actions[1:]

def batch_visual_inference(images):
    return_states = []
    for idx in range(0,len(images),50):
        batch_images = images[idx:idx+50]
        batch_states = []
        for t in range(len(batch_images[0])):
            input_images = [bi[t] for bi in batch_images]
            states = inferencer.inference_batch(input_images)
            # convert to the space used for training model.
            states[:,:,0]=0.5-states[:,:,0]
            # left-right align. (this is due to model overfitting..)
            if t==0:
                for i,s in enumerate(states):
                    if s[0,0] > s[-1,0]:
                        states[i]=s[::-1,:]
            else:
                for i,(s,s0) in enumerate(zip(states,batch_states[0])):
                    if np.linalg.norm(s-s0) > np.linalg.norm(s[::-1,:]-s0):
                        states[i]=s[::-1,:]
            batch_states.append(states)
        batch_states = np.array(batch_states).transpose((1,0,2,3))
        return_states.append(batch_states)
    return_states = np.concatenate(return_states, axis=0)
    return return_states

def load_real_sequence(dir):
    actions = np.load(os.path.join(dir, 'actions.npy'))
    actions = actions[:,[12,13,16,17]]
    num_actions = len(actions)
    states = []
    for i in range(num_actions+1):
        image = Image.open(os.path.join(dir, 'image_%d.png'%(i)))
        state = inferencer.inference(np.array(image))
        states.append(state)

    converted_actions = []
    for act,state  in zip(actions, states):
        dists = np.linalg.norm(state-act[:2], axis=-1)
        action_node = np.argmin(dists)
        action = (action_node, np.array([act[2], act[3]]))
        converted_actions.append(action)

    return states, converted_actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', help='path to load model')
#    parser.add_argument('--gin_config', default='default.gin', help="path to gin config file.")
#    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

#    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)

    simulator = neural_sim('LSTM', args.pretrained_model)

    # find sequences with length 100.
    train_index = []
    for i in range(9000):
        files = glob.glob('/scr1/mengyuan/data/sim3d_sequence/%04d/%04d_*.txt'%(i,i))
        if len(files)>100:
            train_index.append(i)
    test_index = []
    for i in range(9000, 10000):
        files = glob.glob('/scr1/mengyuan/data/sim3d_sequence/%04d/%04d_*.txt'%(i,i))
        if len(files)>100:
            test_index.append(i)
    print(len(train_index), len(test_index))

    start_time = time.time()
    batch_input_states, batch_actions, batch_gt_states = [], [], []
    for i in train_index[:200]:
        states, actions = \
            load_sim_sequence('/scr1/mengyuan/data/sim3d_sequence/%04d'%(i), use_vision=True)
        batch_gt_states.append(states[:59])
        batch_actions.append(actions[:58])
        batch_input_states.append(states[0])

#    batch_gt_images = batch_gt_states.copy() # for debuging
#    batch_gt_states = batch_visual_inference(batch_gt_states)
    batch_gt_states = np.load('trainset_infered_states.npy')
    batch_input_states = batch_gt_states[:,0,:,:]

    batch_gt_gt_states = []
    for i in train_index[:200]:
        states, actions = \
            load_sim_sequence('/scr1/mengyuan/data/sim3d_sequence/%04d'%(i), use_vision=False)
        batch_gt_gt_states.append(states[:59])

    dists = np.linalg.norm(np.array(batch_gt_gt_states)-np.array(batch_gt_states), axis=-1)
    print(np.amax(dists), np.mean(dists))
    pdb.set_trace()

#    states, actions = \
#        load_real_sequence('/scr1/mengyuan/data/real_rope_ours_2/seq_m1_2')

    print("load time: ", time.time()-start_time)
    start_time = time.time()

    batch_gen_states = [batch_input_states]
    state = batch_input_states
    for i in range(58):
        action = [[ac[i]] for ac in batch_actions]
        state = simulator.execute_batch(state, action)
        batch_gen_states.append(state)
    batch_gen_states = np.array(batch_gen_states).transpose((1,0,2,3))
    print("prediction time: ", time.time()-start_time)
    start_time = time.time()

    avg_dists, max_dists = [], []
    for gen_states, states in zip(batch_gen_states, batch_gt_gt_states):
        avg_d, max_d = [], []
        for pred, gt in zip(gen_states, states):
            dists = np.linalg.norm(pred-gt, axis=-1)
            avg_d.append(np.mean(dists))
            max_d.append(np.amax(dists))
        avg_dists.append(avg_d)
        max_dists.append(max_d)
    print("eval time: ", time.time()-start_time)

    plt.figure() # without offseting the start dists.
    plt.plot(np.mean(avg_dists, axis=0), c='C0')
    plt.fill_between(np.arange(59),
                     np.mean(avg_dists, axis=0)-np.std(avg_dists, axis=0),
                     np.mean(avg_dists, axis=0)+np.std(avg_dists, axis=0),
                     alpha = 0.3, facecolor='C0')
    plt.plot(np.mean(max_dists, axis=0), c='C1')
    plt.fill_between(np.arange(59),
                     np.mean(max_dists, axis=0)-np.std(max_dists, axis=0),
                     np.mean(max_dists, axis=0)+np.std(max_dists, axis=0),
                     alpha = 0.3, facecolor='C1')
    plt.axis([0,60,0,0.5])
    plt.savefig('eval_sim_trainset_withvision.png')
    np.savez('eval_sim_trainset_withvision.npz', mean_avg=np.mean(avg_dists, axis=0),
                                     std_avg=np.std(avg_dists, axis=0),
                                     mean_max=np.mean(max_dists, axis=0),
                                     std_max=np.std(max_dists, axis=0))

