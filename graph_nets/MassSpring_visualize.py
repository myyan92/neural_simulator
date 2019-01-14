import time
from build_dataset import data_parser
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets.demos import models
from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
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
num_processing_steps = 8

# Data / training parameters.
batch_size = 1
step_size = 1.0 / 8
eval_dataset = '../neuralsim_test_s1ka10.tfrecords'

# Create the model.
model = models.EncodeProcessDecode(node_output_size=2)

# Base graph
static_graph = [base_graph(128, 0.1) for _ in range(batch_size)]
base_graph = utils_tf.data_dicts_to_graphs_tuple(static_graph)

# create TensorFlow Dataset objects
val_data = tf.data.TFRecordDataset(eval_dataset)
val_data = val_data.map(data_parser)
val_data = val_data.batch(batch_size, drop_remainder=True)
# create TensorFlow Iterator object
iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                           val_data.output_shapes)
start, action, result = iterator.get_next()
# create two initialization ops to switch between the datasets
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


output_ops, _ = roll_out_physics(
    lambda x: model(x, 1), input_graph, num_processing_steps, step_size)

input_graph = make_all_runnable_in_session(input_graph)
output_ops = make_all_runnable_in_session(output_ops)
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, './model-l8')

sess.run(validation_init_op)
i=0
while True:
    try:
        test_values = sess.run({
            "input": input_graph,
            "true_rollout": result,
            "predicted_rollout": output_ops,
            })
        plt.figure()
        input_val = test_values['input'][0].nodes
        gt = test_values['true_rollout']
        pred = test_values['predicted_rollout'][0].nodes
        # pdb.set_trace()
        plt.plot(gt[...,0].flatten(), gt[...,1].flatten())
        plt.plot(pred[:,0], pred[:,1])
        # plt.show()
        i += 1
        plt.savefig('vis_%d.png'%(i))
        print(i)
    except tf.errors.OutOfRangeError:
        break

"""f
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
