import matplotlib
import numpy as np
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

DVF_data_size = [1.0, 0.1]
DVF_avg_dists = [0.0072, 0.0082]
DVF_max_dists = [0.0127, 0.0137]

neural_data_size = [0.01, 0.033, 0.1, 0.25, 0.5, 1.0]
neural_avg_dists = [0.0028, 0.0023, 0.0022, 0.0022, 0.0021, 0.0020]
neural_max_dists = [0.0077, 0.0066, 0.0064, 0.0063, 0.0062, 0.0060]

plt.semilogx(DVF_data_size, DVF_avg_dists, c='C1', marker='.', markersize=10, label='DVF, average')
plt.semilogx(DVF_data_size, DVF_max_dists, c='C1', linestyle='--', marker='.', markersize=10, label='DVF, maximum')
plt.semilogx(neural_data_size, neural_avg_dists, c='C0', marker='.', markersize=10, label='ours, average')
plt.semilogx(neural_data_size, neural_max_dists, c='C0', linestyle='--', marker='.', markersize=10, label='ours, maximum')
plt.ylim([0,0.02])
plt.yticks(np.linspace(0,0.02,5))
plt.xlabel('Relative dataset size')
plt.ylabel(r'$\|S_{pred} - S_{gt}\| (m)$')
plt.legend()
plt.tight_layout()
plt.show()

