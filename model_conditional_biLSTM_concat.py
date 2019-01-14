import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import pdb

class ConditionalLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
  """Taken the kernal and bias from outside so it can be conditioned on inputs.
  """

  def __init__(self,
               num_units,
               kernel, bias,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    """Initialize the conditional LSTM cell."""
    super(ConditionalLSTMCell, self).__init__(
        num_units=num_units, forget_bias=forget_bias, state_is_tuple=state_is_tuple,
        activation=activation, reuse=reuse, name=name, dtype=dtype, **kwargs)
    
    self._kernel = kernel
    self._bias = bias

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    h_depth = self._num_units
    if len(self._kernel.shape) != 3:
      raise ValueError("Expected kernel to be rank 2")
    if self._kernel.shape[1] is None or \
        self._kernel.shape[1] != input_depth + h_depth:
      raise ValueError("Expected kernel's dimension 1 to be %d" % (input_depth+h_depth))
    if self._kernel.shape[2] is None or \
        self._kernel.shape[2] != 4*self._num_units:
      raise ValueError("Expected kernel's dimension 2 to be %d" % (4*self._num_units))
    if len(self._bias.shape) != 2:
      raise ValueError("Expected bias to be rank 1")
    if self._bias.shape[1] is None or \
        self._bias.shape[1] != 4*self._num_units:
      raise ValueError("Expected bias' dimension 1 to be %d" % (4*self._num_units))

    self.built = True

  def call(self, inputs, state):

    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

    concat_inputs = array_ops.concat([inputs, h], 1)
    concat_inputs = array_ops.expand_dims(concat_inputs, 1)
    gate_inputs = math_ops.matmul(concat_inputs, self._kernel)
    gate_inputs = array_ops.squeeze(gate_inputs, axis=1)
    gate_inputs = gate_inputs + self._bias

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=1)

    forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state


class Model:
    def __init__(self):
        self.scope='sim'

    def build(self, input=None, action=None, param=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """

        with tf.variable_scope(self.scope):
            if input is not None and action is not None and param is not None:
                self.input = input
                self.action = action
                self.param = param
            else:
                self.input = tf.placeholder(dtype=tf.float32, shape=[None,128,2])
                self.action = tf.placeholder(dtype=tf.float32, shape=[None,128,2])
                self.param = tf.placeholder(dtype=tf.float32, shape=[None, 1])

            self.ind = tf.norm(self.action, axis=2, keep_dims=True) > 0
            self.ind = tf.cast(self.ind, tf.float32)
            self.concat = tf.concat([self.input, self.action, self.ind], axis=2)

            param_feature = tf.concat([self.param, self.param**2, self.param**3, self.param**4], axis=-1)
            kernel_shape = [37, 128]
            self.cell_kernel = self.dense(param_feature, "conditional_kernel",
                                          channels=np.prod(kernel_shape), activation=None)
            self.cell_kernel = tf.reshape(self.cell_kernel, [-1]+kernel_shape)
            self.cell_bias = self.dense(param_feature, "conditional_bias",
                                        channels=128, activation=None)
            cell = ConditionalLSTMCell(32, self.cell_kernel, self.cell_bias,
                                       forget_bias=1.0, activation=tf.nn.relu6)  # default is tanh
            self.biLSTM, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, self.concat,
                                                             dtype = tf.float32, time_major=False)
            # self.biLSTM stores (hidden_fw, hidden_bw)
            self.feature = tf.concat([self.biLSTM[0], self.biLSTM[1], self.input], axis=2)
            self.pred = self.conv_layer(self.feature, name='pred', channels=2, kernel=1, activation=None)

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
