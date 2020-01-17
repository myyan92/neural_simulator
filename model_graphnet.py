import numpy as np
import tensorflow as tf
from neural_simulator.graphnet import frame, frame_v2
from neural_simulator.graphnet_blocks import two_branch_mlp, hook_law, force_integration
import gin
import pdb

@gin.configurable
class Model:
    def __init__(self, edge_func, node_func, edge_param):
        self.scope='sim'
        self.edge_func = edge_func
        self.node_func = node_func
        self.edge_param = edge_param

    def build(self, input=None, action=None):

        with tf.variable_scope(self.scope):
            if input is not None and action is not None:
                self.input = input
                self.action = action
            else:
                self.input = tf.placeholder(dtype=tf.float32, shape=[None,64,2])
                self.action = tf.placeholder(dtype=tf.float32, shape=[None,64,2])

            self.pred, self.pred_history = frame_v2(input, action, self.edge_param,
                                                    edge_func = self.edge_func,
                                                    node_func = self.node_func)

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)

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
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=self.input.shape)
        self.loss = tf.nn.l2_loss(self.gt_pred-self.pred, name="loss")
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

if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['./model_graphnet.gin'], [])
    model = Model()
    input = tf.placeholder(dtype=tf.float32, shape=(None, 64,2))
    action = tf.placeholder(dtype=tf.float32, shape=(None, 64,2))
    model.build(input, action)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 64,2))
    model.setup_optimizer(0.001, output)
