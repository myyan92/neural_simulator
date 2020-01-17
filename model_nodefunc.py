import numpy as np
import tensorflow as tf
from graphnet_blocks import two_branch_mlp
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

            B, P = tf.shape(self.input)[0], self.input.shape[1]
            index_1 = tf.range(B)
            index_2 = tf.range(P-1)
            index_1, index_2 = tf.meshgrid(index_1, index_2, indexing='ij')
            index_1, index_2 = tf.reshape(index_1, [-1]), tf.reshape(index_2, [-1])
            index_3 = index_2 + 1
            index = tf.stack([index_1, index_2, index_3], axis=-1)
            index_reverse = tf.stack([index_1, index_3, index_2], axis=-1)
            edge_index = tf.concat([index, index_reverse], axis=0)
            edge_weight = tf.ones([B*(P-1)*2,])

            sender_index = edge_index[:,:2]
            receiver_index = tf.gather(edge_index, tf.constant([0,2]), axis=-1)

            diffs = self.action[:,:-1,:] - self.action[:,1:,:]
            edge_feature = tf.reshape(diffs, [-1,2])
            edge_feature = tf.concat([edge_feature, -edge_feature], axis=0)
            edge_feature = edge_feature * tf.expand_dims(edge_weight, axis=1)
            C = edge_feature.shape[-1]

            aggregate_receiver_index = receiver_index[:,0]*P + receiver_index[:,1]
            aggregated_edge_feature = tf.unsorted_segment_sum(edge_feature, aggregate_receiver_index,
                                                              num_segments = B*P)
            aggregated_edge_feature = tf.reshape(aggregated_edge_feature, [-1, P, C])
            self.pred = self.node_func(aggregated_edge_feature, self.input)

            gt_action = self.action[:,2:,:] + self.action[:,:-2,:] - 2*self.action[:,1:-1,:]
            gt_action = tf.concat([self.action[:,1:2,:]-self.action[:,0:1,:],
                                   gt_action,
                                   self.action[:,-2:-1,:]-self.action[:,-1:,:]], axis=1)
            print(self.input.shape)
            print(gt_action.shape)
            self.gt = self.input * 0.9 + gt_action * 0.05
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

    def setup_optimizer(self, learning_rate, GT_pred):
        self.loss = tf.nn.l2_loss(self.gt-self.pred, "loss")  # will broadcast and train on every substep prediction
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99, epsilon=1e-6).minimize(self.loss)
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
