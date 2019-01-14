import numpy as np
import tensorflow as tf

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

            self.fc1 = self.dense(tf.layers.flatten(self.input), 'fc1', 1024, tf.nn.relu)
            self.fc2 = self.dense(self.fc1, 'fc2', 1024, tf.nn.relu)
            self.fc3 = self.dense(self.fc2, 'fc3', 256, tf.nn.relu)
            self.added = self.fc3 + tf.layers.flatten(self.action)
            self.fc4 = self.dense(self.added, 'fc4', 1024, tf.nn.relu)
            self.fc5 = self.dense(self.fc4, 'fc5', 1024, tf.nn.relu)
            self.fc6 = self.dense(self.fc5, 'fc6', 256, None)
            self.pred = tf.reshape(self.fc6, [-1,128,2])

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)


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
