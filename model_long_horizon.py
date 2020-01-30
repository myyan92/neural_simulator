import numpy as np
import tensorflow as tf
from topology.state_2_topology import find_intersections
import pdb, time

class Model:
    def __init__(self):
        self.scope='sim'

    def build_onestep(self, input, action):

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            ind = tf.norm(action, axis=2, keepdims=True) > 0
            ind = tf.cast(ind, tf.float32)
            inputs_concat = tf.concat([input, action, ind], axis=2)
            cell = tf.nn.rnn_cell.GRUCell(512, activation=tf.nn.relu6, name='gru_cell')
            biLSTM_1, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs_concat,
                                                             dtype = tf.float32, time_major=False)
            # biLSTM stores (hidden_fw, hidden_bw)
            feature_1 = tf.concat([biLSTM_1[0], biLSTM_1[1], input], axis=2)

            # generating attention weight matrix
            dists = tf.reduce_sum(input*input, axis=-1)
            dists = tf.expand_dims(dists, axis=-1) + tf.expand_dims(dists, axis=-2) - 2*tf.matmul(input, input, transpose_b=True)
            dists = tf.sqrt(tf.maximum(dists,1e-8))
            segment_lengths = tf.linalg.band_part(dists, 0, 1) - tf.linalg.band_part(dists, 0, 0)
            avg_seg_lengths = tf.reduce_sum(segment_lengths, axis=[1,2], keepdims=True) / tf.cast(input.shape[1]-1, tf.float32)
            attention_w = tf.exp(-dists/avg_seg_lengths)
            attention_w = tf.where(attention_w > tf.exp(-1.2), attention_w, tf.zeros_like(attention_w))
            attention_w = attention_w - tf.linalg.band_part(attention_w, 1, 1)
            attention_w = attention_w / (tf.reduce_sum(attention_w, axis=-1, keepdims=True) + 1e-8)
            attention_feature = tf.matmul(attention_w, feature_1)
            combined_feature = tf.concat([feature_1, attention_feature], axis=2)

            # fc1 = self.dense(combined_feature, 'fc1', 512, 'relu')
            fc2 = self.dense(combined_feature, 'fc2', 1536, 'relu')
            fc2_drop = tf.nn.dropout(fc2, 0.5)
            cell2 = tf.nn.rnn_cell.GRUCell(512, activation=tf.nn.relu6, name='gru_cell_2')
            biLSTM_2, _ = tf.nn.bidirectional_dynamic_rnn(cell2, cell2, fc2_drop,
                                                          dtype = tf.float32, time_major=False)
            feature_2 = tf.concat([biLSTM_2[0], biLSTM_2[1], input], axis=2)
            fc3 = self.dense(feature_2, 'fc3', 768, 'relu')
            # fc3_drop = tf.nn.dropout(fc3, 0.5)

            # cell3 = tf.nn.rnn_cell.GRUCell(256, activation=tf.nn.relu6, name='gru_cell_3')
            # biLSTM_3, _ = tf.nn.bidirectional_dynamic_rnn(cell3, cell3, fc3, dtype=tf.float32, time_major=False)
            # feature_3 = tf.concat([biLSTM_3[0], biLSTM_3[1], input], axis=2)
            # feature_3_drop = tf.nn.dropout(feature_3, 0.5)

            fc4 = self.dense(fc3, 'fc4', 256, 'relu')
            # fc5 = self.dense(fc4, 'fc5', 64, 'relu')
            pred = self.dense(fc4, 'pred', 3, activation=None)

        return pred, [feature_1, attention_feature, feature_2]


    def build(self, steps, input, action):
        self.steps = steps
        with tf.variable_scope(self.scope):
            if input is not None and action is not None:
                self.input = input
                self.action = action
            else:
                self.input = tf.placeholder(dtype=tf.float32, shape=[None,steps,64,3])
                self.action = tf.placeholder(dtype=tf.float32, shape=[None,steps,64,3])

        self.preds = []
        self.debug_layers = []
        for i in range(steps):
            if i==0:
                pred, debug_layer = self.build_onestep(self.input[:,0], self.action[:,0])
                self.preds.append(pred)
                self.debug_layers.append(debug_layer)
            else:
                pred, debug_layer = self.build_onestep(self.preds[-1], self.action[:,i])
                self.preds.append(pred)
                self.debug_layers.append(debug_layer)

        self.pred = tf.stack(self.preds, axis=1)
        self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=1000000)

    def conv_layer(self, bottom, name, channels, kernel=3, stride=1, activation=tf.nn.relu):
        with tf.variable_scope(name):
            k_init = tf.variance_scaling_initializer()
            b_init = tf.zeros_initializer()
            output = tf.layers.conv1d(bottom, channels, kernel_size=kernel, strides=stride, padding='SAME',
                                      activation=activation, kernel_initializer=k_init, bias_initializer=b_init)
        return output

    def dense(self, bottom, name, channels, activation):
        with tf.variable_scope(name):
            k_init = tf.variance_scaling_initializer()
            b_init = tf.zeros_initializer()
            output = tf.layers.dense(bottom, channels, activation=activation,
                                     kernel_initializer=k_init, bias_initializer=b_init)
        return output

    def predict_single(self, sess, input, action):
        pred, = sess.run([self.pred], feed_dict={self.input:input[None], self.action:action[None]})
        return pred[0]

    def predict_batch(self, sess, inputs, actions):
        pdb.set_trace()
        pred, = sess.run([self.pred], feed_dict={self.input:inputs, self.action:actions})
        return pred

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_optimizer(self, learning_rate, GT_position, gather_index_over, gather_index_under):
        if GT_position is not None and gather_index_over is not None and gather_index_under is not None:
            self.gt_pred = GT_position
            self.gather_index_over = gather_index_over
            self.gather_index_under = gather_index_under
        else:
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=[None, self.steps, 64,3])
            self.gather_index_over = tf.placeholder(name="reg_index_over", dtype=tf.int32, shape=[None, None, 4])
            self.gather_index_under = tf.placeholder(name="reg_index_under", dtype=tf.int32, shape=[None, None, 4])

        self.loss = tf.nn.l2_loss(self.gt_pred-self.pred, "loss")
        vecs = self.pred[:,:,1:]-self.pred[:,:,:-1]
        self.reg_loss = tf.reduce_sum(vecs[:,:,1:]*vecs[:,:,:-1])
        pred_pos_over = tf.gather_nd(self.pred, self.gather_index_over)
        pred_pos_under = tf.gather_nd(self.pred, self.gather_index_under)
        self.topo_reg_loss = tf.reduce_sum(tf.nn.relu(pred_pos_under-pred_pos_over+0.005))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss-0.3*self.reg_loss+0.005*self.topo_reg_loss)
        tf.summary.scalar('loss', self.loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, actions, annos):

        intersections = [[find_intersections(GT_pos) for GT_pos in GT_seq] for GT_seq in annos]
        gather_index_over = [ [b,t,it[0]] if it[2]==1 else [b,t,it[1]]
                              for b,intersect_seq in enumerate(intersections)
                              for t,intersect in enumerate(intersect_seq)
                              for it in intersect]
        gather_index_under = [ [b,t,it[1]] if it[2]==1 else [b,t,it[0]]
                              for b,intersect_seq in enumerate(intersections)
                              for t,intersect in enumerate(intersect_seq)
                              for it in intersect]
        gather_index_over=np.array(gather_index_over)[:,:,np.newaxis]
        gather_index_over = np.tile(gather_index_over, (1,1,4))
        gather_index_over[:,2,:] += np.array([-1,0,1,2])
        gather_index_under=np.array(gather_index_under)[:,:,np.newaxis]
        gather_index_under = np.tile(gather_index_under, (1,1,4))
        gather_index_under[:,2,:] += np.array([-1,0,1,2])
        gather_index_over = np.tile(gather_index_over[:,:,:,np.newaxis], (1,1,1,4))
        gather_index_over = gather_index_over.transpose((0,2,3,1)).reshape((-1,3))
        gather_index_under = np.tile(gather_index_under[:,:,np.newaxis,:], (1,1,4,1))
        gather_index_under = gather_index_under.transpose((0,2,3,1)).reshape((-1,3))
        gather_index_over = np.insert(gather_index_over, 3, 2, axis=1)
        gather_index_under = np.insert(gather_index_under, 3, 2, axis=1)

        _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.input:inputs,
                                                                   self.action:actions,
                                                                   self.gt_pred:annos,
                                                                   self.gather_index_over:gather_index_over,
                                                                   self.gather_index_under:gather_index_under})
        return loss

    def save(self, sess, file_dir, step):
        self.saver.save(sess, file_dir, global_step=step)

    def load(self, sess, snapshot):
        self.saver.restore(sess, snapshot)



if __name__=="__main__":
    model = Model()
    input_tf = tf.placeholder(tf.float32, shape=[4,10,64,3])
    action_tf = tf.placeholder(tf.float32, shape=[4,10,64,3])
    gt_tf = tf.placeholder(tf.float32, shape=[4,10,64,3])
    model.build(10, input_tf, action_tf)
    model.setup_optimizer(0.001, gt_tf)
    input_val = np.random.uniform(size=(4,10,64,3))
    action_val = np.random.uniform(size=(4,10,64,3))
    gt_val = np.random.uniform(size=(4,10,64,3))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    pred = model.predict_batch(sess, input_val, action_val)
    model.fit(sess, input_val, action_val, gt_val)
