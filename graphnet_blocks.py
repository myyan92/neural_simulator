import tensorflow as tf
import gin
import pdb

@gin.configurable
def two_branch_mlp(feature_1, feature_2,
              branch1_hidden_units = [],
              branch2_hidden_units = [],
              merge_hidden_units = [],
              output_unit=2,
              batchnorm=True, output_scale=1.0, name='mlp', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        net_1 = feature_1
        for units in branch1_hidden_units:
            net_1 = tf.layers.dense(net_1, units)
            if batchnorm:
                net_1 = tf.layers.batch_normalization(net_1)
            net_1 = tf.nn.tanh(net_1)

        net_2 = feature_2
        for units in branch2_hidden_units:
            net_2 = tf.layers.dense(net_2, units)
            if batchnorm:
                net_2 = tf.layers.batch_normalization(net_2)
            net_2 = tf.nn.tanh(net_2)

        merge_net = tf.concat([net_1, net_2], axis=-1)
        for units in merge_hidden_units:
            merge_net = tf.layers.dense(merge_net, units)
            if batchnorm:
                merge_net = tf.layers.batch_normalization(merge_net)
            merge_net = tf.nn.tanh(merge_net)

        output = tf.layers.dense(merge_net, output_unit,
                                 kernel_initializer=tf.variance_scaling_initializer(scale=output_scale))
    return output

@gin.configurable
def hook_law(aggre_node_feature, edge_attribute, reuse=None, name='hook'):
    with tf.variable_scope(name, reuse=reuse):
        delta_pos,_ = tf.split(aggre_node_feature, 2, axis=-1)
        length = tf.norm(delta_pos, axis=-1, keepdims=True)
        length = tf.maximum(length, 1e-4)
        force = delta_pos / length * (length-edge_attribute)
        youngs = tf.get_variable('youngs', initializer=tf.constant([1.0]),
                                 trainable=True)
        return force*youngs

@gin.configurable
def force_integration(aggre_edge_force, node_state, reuse=None, name='integrator'):
    with tf.variable_scope(name, reuse=reuse):
        node_position, node_velocity = tf.split(node_state, 2, axis=-1)
        inverse_mass = tf.get_variable('inv_mass', initializer=tf.constant([0.01]),
                                        trainable=True)
        inertia = tf.get_variable('inertia', initializer=tf.constant([0.95]),
                                   trainable=True)
        return inertia * node_velocity + inverse_mass * aggre_edge_force

if __name__ == "__main__":

    import numpy as np

    feature1 = tf.placeholder(shape=[None, None, 6], dtype=tf.float32, name='feature1')
    feature2 = tf.placeholder(shape=[None, None, 4], dtype=tf.float32, name='feature2')
    output = two_branch_mlp(feature1, feature2,
                            branch1_hidden_units=[5,5],
                            branch2_hidden_units=[8,8],
                            merge_hidden_units=[3,5])
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    input1 = np.random.rand(8,16,6)
    input2 = np.random.rand(8,16,4)
    result = sess.run(output, feed_dict={feature1:input1, feature2:input2})
    print(result.shape)

