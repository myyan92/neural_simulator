import numpy as np
import tensorflow as tf
import gin

@gin.configurable
class Model_MLP:
    def __init__(self, dim=2):
        self.scope='sim'
        self.dim = dim

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
                self.input = tf.placeholder(dtype=tf.float32, shape=[None, 65, self.dim])
                self.action = tf.placeholder(dtype=tf.float32, shape=[None, 65, self.dim])

            self.ind = tf.norm(self.action, axis=2, keep_dims=True) > 0
            self.ind = tf.cast(self.ind, tf.float32)
            self.concat = tf.concat([self.input, self.action, self.ind], axis=2)
            self.fc1 = self.conv_layer(self.concat, 'fc1', 32, kernel=129, activation=tf.nn.softsign)
            self.fc2 = self.conv_layer(self.fc1, 'fc2', 32, kernel=129, activation=tf.nn.softsign)
            self.fc3 = self.conv_layer(self.fc2, 'fc3', self.dim, kernel=129, scale=0.3, activation=None)
            self.pred = self.input + self.fc3
            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)


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
            self.gt_pred = tf.placeholder(name="gt_pred", dtype=tf.float32, shape=[None, 65, self.dim])
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
    model = Model_MLP(2)
    input = tf.placeholder(dtype=tf.float32, shape=(None, 65,2))
    action = tf.placeholder(dtype=tf.float32, shape=(None, 65,2))
    model.build(input, action)
    output = tf.placeholder(dtype=tf.float32, shape=(None, 65,2))
    model.setup_optimizer(0.001, output)
