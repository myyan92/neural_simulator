import tensorflow as tf
import numpy as np
import gin
from graphnet import substep


def test_node_func(edge_features, node_features):
    return tf.maximum(edge_features, node_features)

def test_edge_func(node_features, edge_features):
    node_features = tf.reduce_max(node_features, axis=-1, keepdims=True)
    return tf.maximum(edge_features, node_features)

edge_index = tf.placeholder(shape=[None,3], dtype=tf.int32, name='index')
edge_weight = tf.placeholder(shape=[None,], dtype=tf.float32, name='weight')
edge_attribute = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='attr')
node_feature = tf.placeholder(shape=[None, 10,1], dtype=tf.float32, name='point')

output = substep(edge_index, edge_weight, node_feature, edge_attribute,
                 edge_func = test_edge_func, node_func = test_node_func, pooling='max')

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# generate random adjacent matrix and edge index
adjacent = np.zeros((10,10))
input_edge_index = []
for _ in range(15):
    pair = np.random.choice(10, size=(2,))
    input_edge_index.append([0,pair[0], pair[1]])
    input_edge_index.append([0,pair[1], pair[0]])
    adjacent[pair[0], pair[1]] = 1
    adjacent[pair[1], pair[0]] = 1
input_edge_index = np.array(input_edge_index)
input_edge_weight = np.ones((input_edge_index.shape[0],))
input_edge_attribute = np.zeros((input_edge_index.shape[0], 1))

input_node_feature = np.zeros((1,10,1))
input_node_feature[0,5] = 1
# comparing tensorflow propagation to graph traversal propagation
for _ in range(10):
    next_node_feature = sess.run(output, feed_dict={edge_index:input_edge_index, edge_weight:input_edge_weight,
                                                    edge_attribute:input_edge_attribute, node_feature:input_node_feature})
    gt_next_node_feature = input_node_feature + adjacent.dot(input_node_feature[0])
    gt_next_node_feature = np.minimum(gt_next_node_feature, 1.0)
    print(next_node_feature[0,:,0])
    print(gt_next_node_feature[0,:,0])
    assert(np.all(next_node_feature[:,:,0:1]==gt_next_node_feature))
    input_node_feature = next_node_feature
