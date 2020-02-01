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
from dynamics_inference.dynamic_models import *
from topology_learning.knot_env import KnotEnv
from povray_render.sample_spline import sample_b_spline, sample_equdistance
from topology.representation import AbstractState
from topology.state_2_topology import state2topology
import matplotlib.pyplot as plt
import os, pdb, time, glob



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', help='path to load model')
#    parser.add_argument('--gin_config', default='default.gin', help="path to gin config file.")
#    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

#    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)

    simulator = neural_sim('GRU_attention', args.pretrained_model)

    data = np.load('../topology_learning/0to1_init_buffers/move-R1_left-1_sign-1_init_buffer.npz')
    start_states = data['obs'][:100]
    action_splines = data['actions'][:100]
#    successes = data['rewards']

    batch_actions = []
    for i,tp in enumerate(action_splines):
      action_node = int(tp[0] * 63)
      action_traj = tp[1:-1]
      height = tp[-1]
      knots = [start_states[i][action_node][:2]]*3 + [action_traj[0:2]] + [action_traj[2:4]]*3
      traj = sample_b_spline(knots)
      traj = sample_equdistance(traj, None, seg_length=0.01).transpose()
      traj_height = np.arange(traj.shape[0]) * 0.01
      traj_height = np.minimum(traj_height, traj_height[::-1])
      traj_height = np.minimum(traj_height, height)
      traj = np.concatenate([traj, traj_height[:,np.newaxis]], axis=-1)
      moves = traj[1:]-traj[:-1]
      actions = [(action_node, m) for m in moves]
      batch_actions.append(actions)

    action_lens = [len(ac) for ac in batch_actions]
    state_trajs = [[] for _ in range(len(batch_actions))]
    states = start_states
    for t in range(np.amax(action_lens)):
        actions = [[ac[t]] if t < action_lens[i] else [(0, np.array([0.0,0.0,0.0]))] for i,ac in enumerate(batch_actions)]
        states = simulator.execute_batch(states, actions, return_traj=False, reset_spring=True)
        for traj,s in zip(state_trajs, states):
            traj.append(s)
    topo_trajs = [[state2topology(ob, update_edges=True, update_faces=False) for ob in traj] for traj in state_trajs]

    end_topo = [tp[l-1][0] for tp,l in zip(topo_trajs, action_lens)]
    desired_topo = AbstractState()
    desired_topo.Reide1(idx=0, left=1, sign=1)
    success = [desired_topo==tp for tp in end_topo]
    pdb.set_trace()


