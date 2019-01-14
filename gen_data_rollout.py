import numpy as np
import os, sys, random, argparse
import glob
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
    exist = glob.glob(os.path.join(args.output_dir, '%04d*.*'%(idx)))
    if len(exist) == 23:
        return

    filename = os.path.join(args.input_dir, '%04d.txt'%(idx))
    samples = np.loadtxt(filename)

    action_node = random.randint(20, samples.shape[0]-20)
    ang = np.random.uniform(0,6.28)
    scale = np.random.uniform(0.05,0.12)
    action = [np.sin(ang)*scale, np.cos(ang)*scale]
    data = rollout_single(samples, action_node, action, 10,
                   " -disable_collisions  -stiffen_bending %d"%(args.bending), return_traj=True)

    for i,d in enumerate(data):
        filename = os.path.join(args.output_dir, '%04d_%d.txt'%(idx, i))
        image = render_samples(d)
        np.savetxt(filename, d)
        im = Image.fromarray(image)
        im.save(filename.replace('.txt','.png'))
    with open(os.path.join(args.output_dir, '%04d_act.txt'%(idx)), 'w') as f:
        f.write('%d %f %f\n' % (action_node, action[0], action[1]))


pool = Pool(6)
idxList = list(range(10000))
pool.map(process_func, idxList)

