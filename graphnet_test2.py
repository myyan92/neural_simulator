import tensorflow as tf
import numpy as np
import gin
from graphnet import frame
import pdb

def test_node_func(edge_features, node_features):
    new_node_y = tf.where(node_features[:,:,1:2]>0.0, tf.zeros_like(edge_features), edge_features)
    return tf.concat([tf.zeros_like(new_node_y), new_node_y], axis=-1)

def test_edge_func(node_features, edge_features):
    node_features = tf.gather(node_features, [1,5], axis=-1)
    node_features = tf.reduce_max(node_features, axis=-1, keepdims=True)
    return node_features

node_feature = tf.placeholder(shape=[None, 10,2], dtype=tf.float32, name='point')
action = tf.placeholder(shape=[None, 10,2], dtype=tf.float32, name='action')
output, history = frame(node_feature, action, edge_param=None,
                        edge_func = test_edge_func, node_func = test_node_func)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# generate random adjacent matrix and edge index
input_node_feature = np.zeros((1,10,2))
input_node_feature[:,:,0] = np.linspace(0,9,10)
input_action = np.zeros((1,10,2))
input_action[0,5,1] = 1.0

# comparing tensorflow propagation to graph traversal propagation
result_output, result_history = sess.run([output,history], feed_dict={node_feature:input_node_feature, action:input_action})
pdb.set_trace()
