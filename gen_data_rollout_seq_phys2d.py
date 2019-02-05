""" Reproduce the way real data are collected.
Each sequence start from one spline curve, and has 100 actions.
Each action range from 0.12-0.36(1cm-3cm), direction around rope normal direction.
Subsequences up to 5 steps can have the same action.
"""

import numpy as np
import os, sys, random, argparse
import glob, time
from physbam_python.rollout_physbam import rollout_single
from multiprocessing import Pool
from PIL import Image
from gen_data.render_samples import render_samples

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', help="directory to store generated data")
parser.add_argument('-i', '--input_dir', 
                    help="optionally get input curves from this directory", required=True)
parser.add_argument('-b', '--bending', type=int,
                    help="Bending stiffness multiplier for physbam", default=10000)
args = parser.parse_args()

#if os.path.isdir(args.output_dir):
#    raise ValueError("WARNING: folder exist, please remove folder or use a new folder name")
#else:
#    os.mkdir(args.output_dir)

def process_func(idx):
    filename = os.path.join(args.input_dir, '%04d.txt'%(idx))
    samples = np.loadtxt(filename)

    count = 0
    actions = []
    start = time.time()
    while count <= 100:
        while True:
            action_node = random.randint(10, samples.shape[0]-10)
            if np.linalg.norm(samples[action_node]) >= 5.3:
                continue
            tangent = samples[action_node+1,:]-samples[action_node-1,:]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([tangent[1], -tangent[0]])
            if np.random.rand()>0.5:
                normal = -normal
            angle = np.random.randn()*0.5
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            move = rotation.dot(normal)
            scale = np.random.uniform(0.03,0.09)
            move = move * scale
            if np.linalg.norm(samples[action_node]+move) >= 5.3:
                continue
            else:
                break
        subseq_length = np.random.randint(3,5)
        for i in range(subseq_length):
            if np.linalg.norm(samples[action_node]+move*(i+1)) > 5.3:
                subsec_length = i

        data = rollout_single(samples, action_node, move, subseq_length*3,
                              " -disable_collisions  -stiffen_bending %d"%(args.bending), return_traj=True)
        print('simulation', time.time()-start)
        if count == 0:
            filename = os.path.join(args.output_dir, '%04d_%d.txt'%(idx, count))
            image = render_samples(samples)
            np.savetxt(filename, samples)
            im = Image.fromarray(image)
            im.save(filename.replace('.txt','.png'))
        for d in data[3::3]:
            count += 1
            filename = os.path.join(args.output_dir, '%04d_%d.txt'%(idx, count))
            image = render_samples(d)
            np.savetxt(filename, d)
            im = Image.fromarray(image)
            im.save(filename.replace('.txt','.png'))
            actions.append((action_node, move))
            print('rendering', count, time.time()-start)
        samples = data[-1]

    with open(os.path.join(args.output_dir, '%04d_act.txt'%(idx)), 'w') as f:
        for action_node, move in actions:
            f.write('%d %f %f\n' % (action_node, move[0]*3, move[1]*3))


pool = Pool(6)
idxList = list(range(125,1000))
pool.map(process_func, idxList)
#process_func(1)
