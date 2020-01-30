import numpy as np
import tensorflow as tf
import pdb

class Model:
    def __init__(self):
        self.scope='sim'

    def build(self, input=None, action=None):

        with tf.variable_scope(self.scope):
            if input is not None and action is not None:
                self.input = input
                self.action = action
            else:
                self.input = tf.placeholder(dtype=tf.float32, shape=[None,64,3])
                self.action = tf.placeholder(dtype=tf.float32, shape=[None,64,3])

            self.ind = tf.norm(self.action, axis=2, keepdims=True) > 0
            self.ind = tf.cast(self.ind, tf.float32)
            self.concat = tf.concat([self.input, self.action, self.ind], axis=2)
            # cell = tf.nn.rnn_cell.LSTMCell(512, forget_bias=1.0, activation=tf.nn.relu6, name='basic_lstm_cell')  # default is tanh
            cell = tf.nn.rnn_cell.GRUCell(512, activation=tf.nn.relu6, name='gru_cell')
            self.biLSTM, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, self.concat,
                                                             dtype = tf.float32, time_major=False)
            # self.biLSTM stores (hidden_fw, hidden_bw)
            self.feature = tf.concat([self.biLSTM[0], self.biLSTM[1], self.input], axis=2)

            # generating attention weight matrix
            dists = tf.reduce_sum(self.input*self.input, axis=-1)
            dists = tf.expand_dims(dists, axis=-1) + tf.expand_dims(dists, axis=-2) - 2*tf.matmul(self.input, self.input, transpose_b=True)
            dists = tf.sqrt(tf.maximum(dists,1e-8))
            segment_lengths = tf.linalg.band_part(dists, 0, 1) - tf.linalg.band_part(dists, 0, 0)
            avg_seg_lengths = tf.reduce_sum(segment_lengths, axis=[1,2], keepdims=True) / tf.cast(self.input.shape[1]-1, tf.float32)
            attention_w = tf.exp(-dists/avg_seg_lengths)
            attention_w = tf.where(attention_w > tf.exp(-1.2), attention_w, tf.zeros_like(attention_w))
            attention_w = attention_w - tf.linalg.band_part(attention_w, 1, 1)
            attention_w = attention_w / (tf.reduce_sum(attention_w, axis=-1, keepdims=True) + 1e-8)
            self.attention_feature = tf.matmul(attention_w, self.feature)
            self.combined_feature = tf.concat([self.feature, self.attention_feature], axis=2)
            # fc1 = self.dense(self.combined_feature, 'fc1', 512, 'relu')
            fc2 = self.dense(self.combined_feature, 'fc2', 1536, 'relu')
            fc2_drop = tf.nn.dropout(fc2, 0.5)
            # cell2 = tf.nn.rnn_cell.LSTMCell(512, forget_bias=1.0, activation=tf.nn.relu6, name='basic_lstm_cell_2')  # default is tanh
            cell2 = tf.nn.rnn_cell.GRUCell(512, activation=tf.nn.relu6, name='gru_cell_2')
            self.biLSTM_2, _ = tf.nn.bidirectional_dynamic_rnn(cell2, cell2, fc2_drop,
                                                               dtype = tf.float32, time_major=False)
            self.feature_2 = tf.concat([self.biLSTM_2[0], self.biLSTM_2[1], self.input], axis=2)
            fc3 = self.dense(self.feature_2, 'fc3', 768, 'relu')
            # fc3_drop = tf.nn.dropout(fc3, 0.5)

            # cell3 = tf.nn.rnn_cell.GRUCell(256, activation=tf.nn.relu6, kernel_initializer=tf.variance_scaling_initializer(), bias_initializer=tf.zeros_initializer())
            # self.biLSTM_3, _ = tf.nn.bidirectional_dynamic_rnn(cell3, cell3, conv1, dtype=tf.float32, time_major=False)
            # self.feature_3 = tf.concat([self.biLSTM_3[0], self.biLSTM_3[1], self.input], axis=2)
            # self.feature_3_drop = tf.nn.dropout(self.feature_3, 0.5)

            fc4 = self.dense(fc3, 'fc4', 256, 'relu')
            # fc5 = self.dense(fc4, 'fc5', 64, 'relu')
            self.pred = self.dense(fc4, 'pred', 3, activation=None)

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
        pred, = sess.run([self.pred], feed_dict={self.input:inputs, self.action:actions})
        return pred

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_optimizer(self, learning_rate, GT_position):
        if GT_position is not None:
            self.gt_pred = GT_position
        else:
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=[None, 64,3])
        self.loss = tf.nn.l2_loss(self.gt_pred-self.pred, "loss")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, actions, annos):
        _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.input:inputs,
                                                                   self.action:actions,
                                                                   self.gt_pred:annos})
        return loss

    def save(self, sess, file_dir, step):
        self.saver.save(sess, file_dir, global_step=step)

    def load(self, sess, snapshot):
        self.saver.restore(sess, snapshot)

