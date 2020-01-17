import time
from build_dataset import data_parser
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets.demos import models
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pdb

def base_graph(n, d):
  """Define a basic mass-spring system graph structure.

  Args:
    n: number of masses
    d: distance between masses (as well as springs' rest length)

  Returns:
    data_dict: dictionary with globals, nodes, edges, receivers and senders
        to represent a structure like the one above.
  """
  # Nodes
  # (position, velocity, indicator fixed)
  nodes = np.zeros((n, 5), dtype=np.float32)

  # Edges.
  # (Stiffness coeff, rest length)
  edges, senders, receivers = [], [], []
  for i in range(n - 1):
    left_node = i
    right_node = i + 1
    edges.append([50., d])
    senders.append(left_node)
    receivers.append(right_node)
    edges.append([50., d])
    senders.append(right_node)
    receivers.append(left_node)

  return {
      "globals": np.array([0.0,0.0]),
      "nodes": nodes,
      "edges": edges,
      "receivers": receivers,
      "senders": senders
  }

def set_rest_lengths(graph):
  """Computes and sets rest lengths for the springs in a physical system.

  The rest length is taken to be the distance between each edge's nodes.

  Args:
    graph: a graphs.GraphsTuple having, for some integers N, E:
        - nodes: Nx5 Tensor of [x, y, _, _, _] for each node.
        - edges: Ex2 Tensor of [spring_constant, _] for each edge.

  Returns:
    The input graph, but with [spring_constant, rest_length] for each edge.
  """
  receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
  sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)
  rest_length = tf.norm(
      receiver_nodes[..., :2] - sender_nodes[..., :2], axis=-1, keepdims=True)
  return graph.replace(
      edges=tf.concat([graph.edges[..., :1], rest_length], axis=-1))

def integrate_to_next_state(input_graph, predicted_graph, step_size):
  # manually integrate velocities to compute new positions
  new_pos = input_graph.nodes[..., :2] + predicted_graph.nodes * step_size
  new_vel = (input_graph.nodes[..., 2:4]*input_graph.nodes[..., 4:5] +
             predicted_graph.nodes *(1.0-input_graph.nodes[..., 4:5]))
  new_nodes = tf.concat(
      [new_pos, new_vel, input_graph.nodes[..., 4:5]], axis=-1) # predicted_graph.nodes,
  return input_graph.replace(nodes=new_nodes)

def roll_out_physics(simulator, graph, steps, step_size):
  """Apply some number of steps of physical laws to an interaction network.

  Args:
    simulator: A SpringMassSimulator, or some module or callable with the same
      signature.
    graph: A graphs.GraphsTuple having, for some integers N, E, G:
        - edges: Nx2 tf.Tensor of [spring_constant, rest_length] for each edge.
        - nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each
          node.
        - globals: Gx2 tf.Tensor containing the gravitational constant.
    steps: An integer.
    step_size: Scalar.

  Returns:
    A pair of:
    - The graph, updated after `steps` steps of simulation;
    - A `steps+1`xNx5 tf.Tensor of the node features at each step.
  """

  def body(t, graph, nodes_per_step):
    predicted_graph = simulator(graph)
    if isinstance(predicted_graph, list):
      predicted_graph = predicted_graph[-1]
    graph = integrate_to_next_state(graph, predicted_graph, step_size)
    return t + 1, graph, nodes_per_step.write(t, graph.nodes)

  nodes_per_step = tf.TensorArray(
      dtype=graph.nodes.dtype, size=steps + 1, element_shape=graph.nodes.shape)
  nodes_per_step = nodes_per_step.write(0, graph.nodes)

  _, g, nodes_per_step = tf.while_loop(
      lambda t, *unused_args: t <= steps,
      body,
      loop_vars=[1, graph, nodes_per_step])
  return g, nodes_per_step.stack()

def create_loss_ops(target_op, output_ops):
  """Create supervised loss operations from targets and outputs.

  Args:
    target_op: The target velocity tf.Tensor.
    output_ops: The list of output graphs from the model.

  Returns:
    A list of loss values (tf.Tensor), one per output op.
  """
  loss_ops = [
      tf.reduce_mean(
          tf.reduce_sum((output_op.nodes - target_op[..., 0:2])**2, axis=-1))
      for output_op in output_ops
  ]
  return loss_ops


def make_all_runnable_in_session(*args):
  """Apply make_runnable_in_session to an iterable of graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


# Model parameters.
num_processing_steps = 16

# Data / training parameters.
batch_size = 64
step_size = 1.0 / num_processing_steps
train_dataset = '../neuralsim_train_s9ka10.tfrecords'
eval_dataset = '../neuralsim_test_s1ka10.tfrecords'

# Create the model.
model = models.EncodeProcessDecode(node_output_size=2)

# Base graph
static_graph = [base_graph(128, 0.1) for _ in range(batch_size)]
base_graph = utils_tf.data_dicts_to_graphs_tuple(static_graph)

# create TensorFlow Dataset objects
tr_data = tf.data.TFRecordDataset(train_dataset)
tr_data = tr_data.map(data_parser)
tr_data = tr_data.shuffle(buffer_size=5000)
tr_data = tr_data.batch(batch_size, drop_remainder=True)
val_data = tf.data.TFRecordDataset(eval_dataset)
val_data = val_data.map(data_parser)
val_data = val_data.batch(batch_size, drop_remainder=True)
# create TensorFlow Iterator object
iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                           tr_data.output_shapes)
start, action, result = iterator.get_next()
# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

# Assigning data fields to the base graph
ind = tf.norm(action, axis=2, keepdims=True) > 0
ind = tf.cast(ind, tf.float32)
concat = tf.concat([start, action, ind], axis=2)
input_graph = base_graph.replace(nodes=tf.reshape(concat, (-1,5)))
input_graph = set_rest_lengths(input_graph)


# target_nodes = tf.reshape(result-start, (-1, 2))
# output_ops = model(input_graph, num_processing_steps) # use num_processing_steps=128
# loss_ops = [tf.reduce_mean(
#             tf.reduce_sum((output_op.nodes - target_nodes)**2, axis=-1))
#             for output_op in output_ops]
# loss_op = tf.add_n(loss_ops)


target_nodes = tf.reshape(result, (-1, 2))
output_ops, _ = roll_out_physics(
    lambda x: model(x, 1), input_graph, num_processing_steps, step_size)
output_ops = output_ops.replace(nodes=output_ops.nodes[...,0:2])
loss_op = tf.reduce_mean(
          tf.reduce_sum((output_ops.nodes - target_nodes)**2, axis=-1))

# Optimizer.
learning_rate = 1e-5
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op)

input_graph = make_all_runnable_in_session(input_graph)
output_ops = make_all_runnable_in_session(output_ops)

saver = tf.train.Saver(max_to_keep=15)
tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=16,
    intra_op_parallelism_threads=16)
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())
saver.restore(sess, 'model-l8')

#debug
tf_vars = tf.trainable_variables()

print("# (epoch number), T (elapsed seconds), "
      "Ltr (training 1-step loss), "
      "Lge (test 1-step loss) ")

for epoch in range(30):
    start_time = time.time()
    losses_tr = []
    losses_ge = []
    last_log_time = start_time
    sess.run(training_init_op)
    while True:
        try:
            train_values = sess.run({
                "step": step_op,
                "loss": loss_op,
                "input_graph": input_graph,
                "target_nodes": target_nodes,
                "outputs": output_ops
            })
            final_w = tf_vars[-2].eval(sess)
            final_b = tf_vars[-1].eval(sess)
#            print(np.amax(np.absolute(final_w)))
#            pdb.set_trace()
            losses_tr.append(train_values["loss"])
        except tf.errors.OutOfRangeError:
            break
    saver.save(sess, './model', global_step=epoch, write_meta_graph=False)

    sess.run(validation_init_op)
    while True:
        try:
            test_values = sess.run({
                "loss": loss_op,
#                "true_rollout": target_nodes,
#                "predicted_rollout": output_ops,
            })
            losses_ge.append(test_values["loss"])
        except tf.errors.OutOfRangeError:
            break

    elapsed = time.time() - start_time
    print("# {:02d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}".format(
        epoch, elapsed, np.mean(losses_tr), np.mean(losses_ge)))


"""f
def get_node_trajectories(rollout_array, batch_size):  # pylint: disable=redefined-outer-name
  return np.split(rollout_array[..., :2], batch_size, axis=1)


fig = plt.figure(1, figsize=(18, 3))
fig.clf()
x = np.array(logged_iterations)
# Next-step Loss.
y = losses_tr
ax = fig.add_subplot(1, 2, 1)
ax.plot(x, y, "k")
ax.set_title("Next step loss")
# Rollout loss.
y = losses_ge
ax = fig.add_subplot(1, 2, 2)
ax.plot(x, y, "k")
ax.set_title("Rollout loss")

# Visualize trajectories.
true_rollouts = get_node_trajectories(test_values["true_rollout"],
                                      batch_size_ge)
predicted_rollouts = get_node_trajectories(test_values["predicted_rollout"],
                                           batch_size_ge)

num_graphs = len(true_rollouts)
num_time_steps = true_rollouts[0].shape[0]

# Plot state sequences.
max_graphs_to_plot = 1
num_graphs_to_plot = min(num_graphs, max_graphs_to_plot)
num_steps_to_plot = 24
max_time_step = num_time_steps - 1
step_indices = np.floor(np.linspace(0, max_time_step,
                                    num_steps_to_plot)).astype(int).tolist()
w = 6
h = int(np.ceil(num_steps_to_plot / w))
fig = plt.figure(101, figsize=(18, 8))
fig.clf()
for i, (true_rollout, predicted_rollout) in enumerate(
    zip(true_rollouts, predicted_rollouts)):
  xys = np.hstack([predicted_rollout, true_rollout]).reshape([-1, 2])
  xs = xys[:, 0]
  ys = xys[:, 1]
  b = 0.05
  xmin = xs.min() - b * xs.ptp()
  xmax = xs.max() + b * xs.ptp()
  ymin = ys.min() - b * ys.ptp()
  ymax = ys.max() + b * ys.ptp()
  if i >= num_graphs_to_plot:
    break
  for j, step_index in enumerate(step_indices):
    iax = i * w + j + 1
    ax = fig.add_subplot(h, w, iax)
    ax.plot(
        true_rollout[step_index, :, 0],
        true_rollout[step_index, :, 1],
        "k",
        label="True")
    ax.plot(
        predicted_rollout[step_index, :, 0],
        predicted_rollout[step_index, :, 1],
        "r",
        label="Predicted")
    ax.set_title("Example {:02d}: frame {:03d}".format(i, step_index))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
      ax.legend(loc=3)

# Plot x and y trajectories over time.
max_graphs_to_plot = 3
num_graphs_to_plot = min(len(true_rollouts), max_graphs_to_plot)
w = 2
h = num_graphs_to_plot
fig = plt.figure(102, figsize=(18, 12))
fig.clf()
for i, (true_rollout, predicted_rollout) in enumerate(
    zip(true_rollouts, predicted_rollouts)):
  if i >= num_graphs_to_plot:
    break
  t = np.arange(num_time_steps)
  for j in range(2):
    coord_string = "x" if j == 0 else "y"
    iax = i * 2 + j + 1
    ax = fig.add_subplot(h, w, iax)
    ax.plot(t, true_rollout[..., j], "k", label="True")
    ax.plot(t, predicted_rollout[..., j], "r", label="Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("{} coordinate".format(coord_string))
    ax.set_title("Example {:02d}: Predicted vs actual coords over time".format(
        i))
    ax.set_frame_on(False)
    if i == 0 and j == 1:
      handles, labels = ax.get_legend_handles_labels()
      unique_labels = []
      unique_handles = []
      for i, (handle, label) in enumerate(zip(handles, labels)):  # pylint: disable=redefined-outer-name
        if label not in unique_labels:
          unique_labels.append(label)
          unique_handles.append(handle)
      ax.legend(unique_handles, unique_labels, loc=3)
"""
