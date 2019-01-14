import numpy as np
import tensorflow as tf
import pdb

class Model:
    def __init__(self):
        self.scope='sim'

    def build(self, input=None, action=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """

        with tf.variable_scope(self.scope):
            if input is not None and action is not None:
                self.input = input
                self.action = action
            else:
                self.input = tf.placeholder(dtype=tf.float32, shape=[None,128,2])
                self.action = tf.placeholder(dtype=tf.float32, shape=[None,128,2])

            self.add = self.input + self.action
            self.ind = tf.norm(self.action, axis=2, keep_dims=True) > 0
            self.ind = tf.cast(self.ind, tf.float32)

            bottom = self.add
            for i in range(128):
                bottom = self.res_block(bottom, 'resblock') # weights are tied
                bottom = bottom * (1-self.ind) + self.add*self.ind  # enforce position constraint
            self.pred = bottom

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)

    def res_block(self, bottom, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            transformer_filter = tf.constant([[[1,0,0,0],[0,1,0,0]],
                                              [[-1,0,-1,0],[0,-1,0,-1]],
                                              [[0,0,1,0], [0,0,0,1]]], dtype=tf.float32)
            bottom_ = tf.pad(bottom, tf.constant([[0,0],[1,1],[0,0]]), mode='SYMMETRIC')
            conv0 = tf.nn.conv1d(bottom_, transformer_filter, stride=1, padding='VALID')
            conv1 = self.conv_layer(conv0, 'conv1', 64, activation=tf.nn.leaky_relu)
            conv2 = self.conv_layer(conv1, 'conv2', 64, kernel=1, activation=tf.nn.leaky_relu)
            conv3 = self.conv_layer(conv2, 'conv3', 64, kernel=1, activation=tf.nn.leaky_relu)
            conv4 = self.conv_layer(conv3, 'conv4', 2, scale=0.01, activation=None)
            output = bottom + conv4
        return output

    def conv_layer(self, bottom, name, channels, kernel=3, stride=1, scale=1.0, activation=tf.nn.relu):
        with tf.variable_scope(name):
            k_init = tf.variance_scaling_initializer(scale)
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
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=[None, 128,2])
        self.loss = tf.nn.l2_loss(self.gt_pred-self.pred, "loss")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, actions, annos):
        _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.input:inputs,
                                                                   self.action:actions,
                                                                   self.gt_pred:annos})
        return loss

    def save(self, sess, step):
        self.saver.save(sess, './model', global_step=step)

    def load(self, sess, snapshot):
        self.saver.restore(sess, snapshot)

if __name__ == "__main__":
    model = Model()
    input = tf.placeholder(dtype=tf.float32, shape=(None, 128,2))
    action = tf.placeholder(dtype=tf.float32, shape=(None, 128,2))
    model.build(input, action)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 128,2))
    model.setup_optimizer(0.001, output)
