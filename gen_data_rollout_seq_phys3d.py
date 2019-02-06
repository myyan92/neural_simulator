""" Reproduce the way real data are collected.
Each sequence start from one spline curve, and has 100 actions.
Each action range from 0.12-0.36(1cm-3cm), direction around rope normal direction.
Subsequences up to 5 steps can have the same action.
"""

import numpy as np
import os, sys, random, argparse
import glob, time
from multiprocessing import Pool
from physbam_python.rollout_physbam_3d import rollout_single
from physbam_python.util import hasCollision, hasShapeAngle

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', help="directory to store generated data")
parser.add_argument('-i', '--input_dir', 
                    help="optionally get input curves from this directory", default='/scr1/ylzhu/neural_simulator/data_with_knots/')
parser.add_argument('-b', '--bending', type=float,
                    help="Bending stiffness multiplier for physbam", default=0.88)#0.218)
parser.add_argument('-l', '--linear', type=float,
                    help="Linear stiffness multiplier for physbam", default=0.7)#2.223)
parser.add_argument('-f', '--friction', type=float,
                    help="friction multiplier for physbam", default=0.166)#76)
args = parser.parse_args()


def process_func(idx):
    filename = os.path.join(args.input_dir, '%04d.txt'%(idx))
    samples = np.loadtxt(filename)[::2,:]/12
    if hasCollision(samples):
        print("!!!collision on", idx)
        return
    if hasSharpAngle(samples):
        print("!!!sharp angle on", idx)
        return
    if os.path.isfile(os.path.join(args.output_dir,'{:08d}_act.txt'.format(idx))):
        print("sim exist", idx)
        return


    count = 0
    restart_num = 0
    actions = []
    target = 100
    while count <= target:

        cnt = 0
        while cnt < 1000:
            cnt+=1
            action_node = random.randint(10, samples.shape[0]-10)
            if np.linalg.norm(samples[action_node]) >= 5.3/12:
                continue
            tangent = samples[action_node+1,:]-samples[action_node-1,:]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([tangent[1], -tangent[0]])
            if np.random.rand()>0.5:
                normal = -normal
            angle = np.random.randn()*0.5
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            move = rotation.dot(normal)
            scale = np.random.uniform(0.01,0.03)
            move = move * scale
            if np.linalg.norm(samples[action_node]+move) >= 5.3/12:
                continue
            else:
                break
        if cnt > 999:
            print('cannot find action',idx,'at',count)
            break

        subseq_length = np.random.randint(3,5)
        for i in range(subseq_length):
           if np.linalg.norm(samples[action_node]+move*(i+1)) > 5.3/12:
               subseq_length = i
        action = np.array([[move[0],move[1],action_node/64.0]]*subseq_length)

        data = rollout_single(samples, action,
            physbam_args=" -stiffen_bending %f -stiffen_linear %f -friction %f -v 0 -dt 1e-3 "\
            %(args.bending, args.linear, args.friction), return_traj=True, keep_files=False,
            output_dir='./Sims/')

        samples = data[-1]
        if count == 0:
            filename = os.path.join(args.output_dir, '%08d_%03d.txt'%(idx, count))
            np.savetxt(filename, samples)
        for d in data:
            if hasCollision(d):
                if count > target//2 and target > 10:
                    target = target//2
                    break
                else:
                    print("has collision on count",count,"/",target,"index",idx,"restart num",restart_num)
                    restart_num+=1
                    if restart_num > 20:
                        return
                    count = 0
                    actions = []
                    start = time.time()
                    filename = os.path.join(args.input_dir, '%04d.txt'%(idx))
                    samples = np.loadtxt(filename)[::2,:]/12
                    break
            count += 1
            filename = os.path.join(args.output_dir, '%08d_%03d.txt'%(idx, count))
            np.savetxt(filename, d)
            actions.append((action_node, move))

    with open(os.path.join(args.output_dir, '%08d_act.txt'%(idx)), 'w') as f:
        for action_node, move in actions:
            f.write('%d %f %f\n' % (action_node, move[0], move[1]))


pool = Pool()
idxList = list(range(3241,10000))
pool.map(process_func, idxList)
# process_func(53)

