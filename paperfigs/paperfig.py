import numpy as np
import matplotlib
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

neural_novision_eval = np.load('eval_sim_testset.npz')
neural_novision_train = np.load('eval_sim_trainset.npz')
neural_vision_eval = np.load('eval_sim_testset_withvision.npz')
neural_vision_train = np.load('eval_sim_trainset_withvision.npz')

baseline_train = np.load('../../baseline-DVF-0/eval_simseq3d_rollout_trainset.npz')
baseline_eval =  np.load('../../baseline-DVF-0/eval_simseq3d_rollout_testset.npz')
baseline_train_avg_dists=baseline_train['avg_dists']-baseline_train['start_avg_dists'].reshape((-1,1))
baseline_train_max_dists=baseline_train['max_dists']-baseline_train['start_max_dists'].reshape((-1,1))
baseline_train = {'mean_avg': np.mean(baseline_train_avg_dists, axis=0),
                  'std_avg': np.std(baseline_train_avg_dists, axis=0),
                  'mean_max': np.mean(baseline_train_max_dists, axis=0),
                  'std_max': np.std(baseline_train_max_dists, axis=0)}
baseline_eval_avg_dists=baseline_eval['avg_dists']-baseline_eval['start_avg_dists'].reshape((-1,1))
baseline_eval_max_dists=baseline_eval['max_dists']-baseline_eval['start_max_dists'].reshape((-1,1))
baseline_eval = {'mean_avg': np.mean(baseline_eval_avg_dists, axis=0), 
                  'std_avg': np.std(baseline_eval_avg_dists, axis=0),
                  'mean_max': np.mean(baseline_eval_max_dists, axis=0),
                  'std_max': np.std(baseline_eval_max_dists, axis=0)}


series = [(neural_vision_eval['mean_avg'], neural_vision_eval['std_avg']),
          (neural_vision_eval['mean_max'], neural_vision_eval['std_max']),
          (baseline_eval['mean_avg'], baseline_eval['std_avg']),
          (baseline_eval['mean_max'], baseline_eval['std_max']), ]

labels = ['ours, average', 'ours, maximum', 'DVF, average', 'DVF, maximum']
colors = ['C0', 'C0', 'C1', 'C1']
linestyles = ['-', '--', '-', '--']

for data, label, c, l in zip(series, labels, colors, linestyles):
    mean = data[0][:50]
    std = data[1][:50]
    plt.plot(mean, c=c, label=label, linestyle=l, linewidth=2.5)
    plt.fill_between(np.arange(len(mean)), mean-std, mean+std, facecolor=c, alpha=0.2)
plt.axis([0, 50, 0, 0.25])
plt.xlabel('Time step (t)')
plt.ylabel(r'$\|S_{rollout} - S_{gt}\| (m)$')
plt.legend(loc=2)
plt.tight_layout()
plt.show()
