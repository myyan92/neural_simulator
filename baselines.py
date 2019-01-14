import numpy as np
import pdb

action_pattern = "/scr-ssd/mengyuan/neural_simulator/action_state/%04d_act.txt"
position_pattern = "/scr-ssd/mengyuan/neural_simulator/action_state/%04d_%d.txt"
num_curve = 100
num_action = 10
starts = []
actions = []
results = []
for i in range(3000):
    for a in range(1,num_action+1):
        result = np.loadtxt(position_pattern % (i,a))
        start = np.loadtxt(position_pattern % (i,0))
        with open(action_pattern % (i)) as f:
            line = f.readline()
        tokens = line.strip().split()
        action_node = int(tokens[0])
        action_x, action_y = float(tokens[1]), float(tokens[2])
        action = np.zeros_like(start)
        action[action_node,:]=np.array([action_x, action_y])
        action *= (a-1)

        starts.append(start)
        actions.append(action)
        results.append(result)

starts = np.array(starts)
actions = np.array(actions)
results = np.array(results)
null_loss = np.sum(np.square(results-starts)) / 2.0 / starts.shape[0] / starts.shape[1]
print("loss predicting zero change: ", null_loss)
one_loss = np.sum(np.square(results-actions-starts)) / 2.0 / starts.shape[0] / starts.shape[1]
print("loss predicting start+action: ", one_loss)

mat_A = np.dot(actions.reshape((-1,256)).transpose(), actions.reshape((-1,256)))
mat_B = np.dot(actions.reshape((-1,256)).transpose(), (results-starts).reshape((-1,256)))
X=np.linalg.lstsq(mat_A, mat_B)
W=X[0]
linear_pred = np.dot(actions.reshape((-1,256)), W).reshape((-1,128,2))
linear_loss = np.sum(np.square(results-starts-linear_pred)) / 2.0 / starts.shape[0] / starts.shape[1]
print("loss with linear model: ", linear_loss)
max_dev = np.mean(np.amax(np.linalg.norm(results-starts-linear_pred, axis=2), axis=1))
print("max deviation with linear model: ", max_dev)

W2=np.zeros_like(W)
for i in range(-255,256):
    s = []
    for j in range(max(20,-i), min(236,256-i)):
        s.append(W[j,j+i])
    if len(s)>0:
        s = np.mean(s)
    else:
        s = 0
    for j in range(max(0,-i), min(256,256-i)):
        W2[j,j+i]=s
linear_pred = np.dot(actions.reshape((-1,256)), W2).reshape((-1,128,2))
linear_loss = np.sum(np.square(results-starts-linear_pred)) / 2.0 / starts.shape[0] / starts.shape[1]
print("loss with linear conv model: ", linear_loss)
max_dev = np.mean(np.amax(np.linalg.norm(results-starts-linear_pred, axis=2), axis=1))
print("max deviation with linear conv model: ", max_dev)

