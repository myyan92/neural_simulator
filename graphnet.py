import tensorflow as tf
import gin
from neural_simulator.graphnet_blocks import two_branch_mlp
import pdb

def substep(edge_index, edge_weight, node_feature, edge_attribute,
            node_func, edge_func, pooling="sum", reuse=None):
    """ Graph network one substep, involving node -> edge -> node
    Arguments:
        edge_index: Nx3 tensor, N is the total number of edges in the batch.
                    The 3 numbers in a row represent B,I,J index of an edge
                    in the adjacent matrix.
        edge_weight: (N,) tensor, the weight of edges.
        edge_attribute: (N,D) tensor, attribute of edges. Constant across one simulation.
        node_feature: BxPxC tensor, where P is the number of nodes in the graph
                      and is fixed across all samples.
        node_func: Tensorflow function that applies to every node feature.
                   Must take aggregated edge feature and node_feature as input and
                   broadcast to the first two dimensions.
        edge_func: Tensorflow function that applies to every edge feature.
                   Must take two 2D tensor as input and broadcast to the first dim.
        pooling: One of "Max", "Sum" and "Mean". How edge features are aggregated
                   to nodes.
    Returns:
        a tensor with shape BxPxC' as new node features.
    """
    sender_index = edge_index[:,:2]
    receiver_index = tf.gather(edge_index, tf.constant([0,2]), axis=-1)
    sender_node_feature = tf.gather_nd(node_feature, sender_index)
    receiver_node_feature = tf.gather_nd(node_feature, receiver_index)
    edge_input_feature = sender_node_feature - receiver_node_feature
    edge_output_feature = edge_func(edge_input_feature, edge_attribute,
                                    name='edge_func', reuse=reuse)  # TODO have different types of edges for spring and friction?
    # weighting edge output feature with edge weights
    edge_output_feature = edge_output_feature * tf.expand_dims(edge_weight, axis=1)

    B, P, C = tf.shape(node_feature)[0], node_feature.shape[1], edge_output_feature.shape[-1]
    aggregate_receiver_index = receiver_index[:,0]*node_feature.shape[1] + receiver_index[:,1]
    if pooling.lower() == 'max':
        aggregated_edge_feature = tf.unsorted_segment_max(edge_output_feature, aggregate_receiver_index,
                                                          num_segments = B*P)
    elif pooling.lower() == 'sum':
        aggregated_edge_feature = tf.unsorted_segment_sum(edge_output_feature, aggregate_receiver_index,
                                                          num_segments = B*P)
    elif pooling.lower() == 'mean':
        aggregated_edge_feature = tf.unsorted_segment_mean(edge_output_feature, aggregate_receiver_index,
                                                           num_segments = B*P)
    else:
        raise NotImplementedError("pooling method %s is not implemented" % (pooling))
    aggregated_edge_feature = tf.reshape(aggregated_edge_feature, [-1, P, C])
    output_node_feature = node_func(aggregated_edge_feature, node_feature,
                                    name='node_func', reuse=reuse)

    return output_node_feature

def substep_v2(edge_index, edge_weight, node_feature, edge_attribute,
               node_func, edge_func, pooling="sum", reuse=None):
    """ Graph network one substep, involving node -> edge -> node
    Arguments:
        edge_index: Nx3 tensor, N is the total number of edges in the batch.
                    The 3 numbers in a row represent B,I,J index of an edge
                    in the adjacent matrix.
        edge_weight: (N,) tensor, the weight of edges.
        edge_attribute: (N,D) tensor, attribute of edges. Constant across one simulation.
        node_feature: BxPxC tensor, where P is the number of nodes in the graph
                      and is fixed across all samples.
        node_func: Tensorflow function that applies to every node feature.
                   Must take aggregated edge feature and node_feature as input and
                   broadcast to the first two dimensions.
        edge_func: Tensorflow function that applies to every edge feature.
                   Must take two 2D tensor as input and broadcast to the first dim.
        pooling: One of "Max", "Sum" and "Mean". How edge features are aggregated
                   to nodes.
    Returns:
        a tensor with shape BxPxC' as new node features.
    """
    edge_output_features, edge_receivers = [], []
    for edge_type, (edge_i, edge_w, edge_a) in enumerate(zip(edge_index, edge_weight, edge_attribute)):
        sender_index = edge_i[:,:2]
        receiver_index = tf.gather(edge_i, tf.constant([0,2]), axis=-1)
        sender_node_feature = tf.gather_nd(node_feature, sender_index)
        receiver_node_feature = tf.gather_nd(node_feature, receiver_index)
        edge_input_feature = sender_node_feature - receiver_node_feature
        edge_output_feature = edge_func(edge_input_feature, edge_a,
                                        name='edge_func_type%d'%(edge_type), reuse=reuse)
        # weighting edge output feature with edge weights
        edge_output_feature = edge_output_feature * tf.expand_dims(edge_w, axis=1)
        edge_output_features.append(edge_output_feature)
        aggregate_receiver_index = receiver_index[:,0]*node_feature.shape[1] + receiver_index[:,1]
        edge_receivers.append(aggregate_receiver_index)

    edge_output_features = tf.concat(edge_output_features, axis=0)
    edge_receivers = tf.concat(edge_receivers, axis=0)
    B, P, C = tf.shape(node_feature)[0], node_feature.shape[1], edge_output_feature.shape[-1]
    if pooling.lower() == 'max':
        aggregated_edge_feature = tf.unsorted_segment_max(edge_output_features, edge_receivers,
                                                          num_segments = B*P)
    elif pooling.lower() == 'sum':
        aggregated_edge_feature = tf.unsorted_segment_sum(edge_output_features, edge_receivers,
                                                          num_segments = B*P)
    elif pooling.lower() == 'mean':
        aggregated_edge_feature = tf.unsorted_segment_mean(edge_output_features, edge_receivers,
                                                           num_segments = B*P)
    else:
        raise NotImplementedError("pooling method %s is not implemented" % (pooling))
    aggregated_edge_feature = tf.reshape(aggregated_edge_feature, [-1, P, C])
    output_node_feature = node_func(aggregated_edge_feature, node_feature,
                                    name='node_func', reuse=reuse)

    return output_node_feature

@gin.configurable
def frame(node_positions, actions, edge_param, node_func, edge_func):
    """ Simulate one action forward.
    This function constructs edge index, weights and attributes from
    node_positions, divide actions as velocity constraints, and
    recursively run substep function.
    Arguments:
        node_positions: BxPx2 or BxPx3 node positions.
        actions: The same shape as node_positions.
        edge_param: A vector representing edges' physical parameters.
                    edge function is conditioned on this edge_param.
        node_func: A tensorflow function that takes aggregated edge feature
                   and node state as input.
        edge_func: A tensorflow function that takes aggregated node
                   feature and edge attribute as input. With additional
                   argument edge_param.
    Return:
        a tensor the same shape as node_positions.
    """
    # Simple case of fixed graph (chain)
    B, P = tf.shape(node_positions)[0], node_positions.shape[1]
    index_1 = tf.range(B)
    index_2 = tf.range(P-1)
    index_1, index_2 = tf.meshgrid(index_1, index_2, indexing='ij')
    index_1, index_2 = tf.reshape(index_1, [-1]), tf.reshape(index_2, [-1])
    index_3 = index_2 + 1
    index = tf.stack([index_1, index_2, index_3], axis=-1)
    index_reverse = tf.stack([index_1, index_3, index_2], axis=-1)
    edge_index = tf.concat([index, index_reverse], axis=0)
    edge_weight = tf.ones([B*(P-1)*2])

    diffs = node_positions[:,:-1,:] - node_positions[:,1:,:]
    edge_attribute = tf.norm(diffs, axis=-1)
    edge_attribute = tf.reshape(edge_attribute, [-1,1])
    edge_attribute = tf.concat([edge_attribute, edge_attribute], axis=0)

    action_velocity = tf.reduce_mean(edge_attribute) / 10.0
#    debug_node_update=tf.ones(shape=[15,8,64,3]) * 1e-2
#    print(debug_node_update)
    def tf_while_condition(node_state, action, node_update, count, history):
        cond_1 = tf.reduce_any( tf.norm(action, axis=[1,2]) > 1e-6 )
        cond_2 = tf.reduce_any( tf.norm(node_update, axis=[1,2]) > 3e-3 )
        return tf.logical_or(cond_1, cond_2)

    def tf_while_body(node_state, action, node_update, count, history):
        # get substep action and substract from action
        action_norm = tf.norm(action, axis=-1, keepdims=True)
        substep_action_norm = tf.minimum(action_norm, action_velocity)
        substep_action = action / (action_norm+1e-6) * substep_action_norm
        action = action - substep_action

        # enforce velocity and simulate
        node_state = tf.stop_gradient(node_state)
        node_position, node_velocity = tf.split(node_state, 2, axis=-1)
        node_position = node_position + substep_action
        node_state = tf.concat([node_position, node_velocity], axis=-1)
        node_update = substep(edge_index, edge_weight, node_state, edge_attribute,
                              node_func, edge_func, pooling='sum')
#        node_update= debug_node_update[count] # to observe gradient propagation of while_loop
        # integrate
        update_mask = tf.norm(substep_action, axis=-1, keepdims=True) > 1e-7
        update_mask = tf.tile(update_mask, tf.stack([1,1,node_position.shape[-1]]) )
        node_update = tf.where(update_mask, tf.zeros_like(node_update), node_update)
        node_position, node_velocity = tf.split(node_state, 2, axis=-1)
        node_position = node_position + node_update
        node_state = tf.concat([node_position, node_update], axis=-1)
        # write node state to history
        history = history.write(count, node_state)
        count = count + 1
        return node_state, action, node_update, count, history

    node_update = tf.zeros_like(node_positions)
    node_state = tf.concat([node_positions, node_update], axis=-1)
    count = 0
    history = tf.TensorArray(dtype=tf.float32, size=4, dynamic_size=True)
    final_node_state, _, _, final_count, final_history = tf.while_loop(tf_while_condition, tf_while_body,
                                               loop_vars = (node_state, actions, node_update, count, history),
                                               maximum_iterations=8)
    final_node_positions, _ = tf.split(final_node_state, 2, axis=-1)
    final_history = final_history.stack()
    return final_node_positions, final_history

def frame_v2(node_positions, actions, edge_param, node_func, edge_func):
    """ Simulate one action forward.
    This function constructs edge index, weights and attributes from
    node_positions, divide actions as velocity constraints, and
    recursively run substep function.
    Arguments:
        node_positions: BxPx2 or BxPx3 node positions.
        actions: The same shape as node_positions.
        edge_param: A vector representing edges' physical parameters.
                    edge function is conditioned on this edge_param.
        node_func: A tensorflow function that takes aggregated edge feature
                   and node state as input.
        edge_func: A tensorflow function that takes aggregated node
                   feature and edge attribute as input. With additional
                   argument edge_param.
    Return:
        a tensor the same shape as node_positions.
    """
    # Simple case of fixed graph (chain)
    B, P = tf.shape(node_positions)[0], node_positions.shape[1]
    index_1 = tf.range(B)
    index_2 = tf.range(P-1)
    index_1, index_2 = tf.meshgrid(index_1, index_2, indexing='ij')
    index_1, index_2 = tf.reshape(index_1, [-1]), tf.reshape(index_2, [-1])
    index_3 = index_2 + 1
    index = tf.stack([index_1, index_2, index_3], axis=-1)
    index_reverse = tf.stack([index_1, index_3, index_2], axis=-1)
    edge_index = tf.concat([index, index_reverse], axis=0)
    edge_weight = tf.ones([B*(P-1)*2])

    diffs = node_positions[:,:-1,:] - node_positions[:,1:,:]
    edge_attribute = tf.norm(diffs, axis=-1)
    edge_attribute = tf.reshape(edge_attribute, [-1,1])
    edge_attribute = tf.concat([edge_attribute, edge_attribute], axis=0)

    index_1 = tf.range(B)
    index_2 = tf.range(P-2)
    index_1, index_2 = tf.meshgrid(index_1, index_2, indexing='ij')
    index_1, index_2 = tf.reshape(index_1, [-1]), tf.reshape(index_2, [-1])
    index_3 = index_2 + 2
    index = tf.stack([index_1, index_2, index_3], axis=-1)
    index_reverse = tf.stack([index_1, index_3, index_2], axis=-1)
    edge_index_bend = tf.concat([index, index_reverse], axis=0)
    edge_weight_bend = tf.ones([B*(P-2)*2])

    edge_attribute_bend = tf.norm(diffs, axis=-1)
    edge_attribute_bend = edge_attribute_bend[:,:-1] + edge_attribute_bend[:,1:]
    edge_attribute_bend = tf.reshape(edge_attribute_bend, [-1,1])
    edge_attribute_bend = tf.concat([edge_attribute_bend, edge_attribute_bend], axis=0)

    num_processing_steps = 128
    node_update = tf.zeros_like(node_positions)
    node_state = tf.concat([node_positions, node_update], axis=-1)
    history_states = []
    for step in range(num_processing_steps):
        if step < 8:
            substep_action = actions / 8.0
        else:
            substep_action = tf.zeros_like(actions)
        control_index = tf.norm(actions, axis=-1, keepdims=True) > 0

        node_position, node_velocity = tf.split(node_state, 2, axis=-1)
        node_state = tf.concat([node_position+substep_action, node_velocity], axis=-1)
        if step == 0:
            node_update = substep_v2([edge_index,edge_index_bend], [edge_weight,edge_weight_bend],
                                     node_state, [edge_attribute,edge_attribute_bend],
                                     node_func, edge_func, pooling='sum')
        else:
            node_update = substep_v2([edge_index,edge_index_bend], [edge_weight,edge_weight_bend],
                                     node_state, [edge_attribute,edge_attribute_bend],
                                     node_func, edge_func, pooling='sum', reuse=True)
        update_mask = tf.tile(control_index, tf.stack([1,1,node_update.shape[-1]]) )
        node_update = tf.where(update_mask, tf.zeros_like(node_update), node_update)
        node_position, node_velocity = tf.split(node_state, 2, axis=-1)
        node_state = tf.concat([node_position+node_update, node_update], axis=-1)
        history_states.append(node_state)

    final_node_positions, _ = tf.split(node_state, 2, axis=-1)
    history_states = tf.stack(history_states, axis=0)
    return final_node_positions, history_states



if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['./graphnet_blocks.gin'], [])
    import numpy as np

    node_position = tf.placeholder(shape=[8, 64,3], dtype=tf.float32, name='node')
    action = tf.placeholder(shape=[8, 64,3], dtype=tf.float32, name='action')
    output, history = frame(node_position, action, edge_param=None)
    debug_loss = tf.nn.l2_loss(output)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    input_node = np.random.rand(8,64,3)
    input_action = np.zeros((8,64,3))
    input_action[:,5,0] = 0.2
    diffs = input_node[:,:-1,:]-input_node[:,1:,:]
    velocity = np.mean(np.linalg.norm(diffs,axis=-1))
    num_steps = np.amax(np.linalg.norm(input_action, axis=-1)) / velocity
    print(num_steps)
    result, result_history = sess.run([output,history], feed_dict={node_position:input_node, action:input_action})
    print(result.shape)
    print(result_history.shape)
    pdb.set_trace()
