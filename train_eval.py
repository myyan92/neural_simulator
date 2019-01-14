import tensorflow as tf
from model_wrapper import Model
from build_dataset import data_parser
import numpy as np
import argparse
import pdb
import matplotlib.pyplot as plt

class Trainner():
    def __init__(self, model_type, train_dataset, eval_dataset, num_epoch, batch_size):

        # create TensorFlow Dataset objects
        tr_data = tf.data.TFRecordDataset(train_dataset)
        tr_data = tr_data.map(data_parser)
        tr_data = tr_data.shuffle(buffer_size=5000)
        tr_data = tr_data.batch(batch_size)
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(batch_size)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                   tr_data.output_shapes)
        self.start, self.action, self.result = iterator.get_next()  # self.next_gradient
        # create two initialization ops to switch between the datasets
        self.training_init_op = iterator.make_initializer(tr_data)
        self.validation_init_op = iterator.make_initializer(val_data)
        
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.num_epoch = num_epoch
        self.model = Model(model_type)  #load pretrained weights
        self.model.build(input=self.start, action=self.action)
        self.model.setup_optimizer(0.001, self.result)
        self.global_step = 0
        self.train_writer = tf.summary.FileWriter('tboard', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self):
#        tensors = []
#        for l in range(128):
#            if l == 0:
#                name = 'sim/resblock/conv4/conv1d/BiasAdd:0'
#            else:
#                name = 'sim/resblock_%d/conv4/conv1d/BiasAdd:0' %(l)
#            tensors.append(tf.get_default_graph().get_tensor_by_name(name))

        self.sess.run(self.training_init_op)
        losses = []
        while True:
            try:
                summary, _, loss = self.sess.run([self.model.merged_summary, self.model.optimizer, self.model.loss])
#                tensorMax = [np.amax(t) for t in tensors_val]
#                stats=(np.mean(tensorMax), np.std(tensorMax))
#                print(stats)
                self.train_writer.add_summary(summary, self.global_step)
                self.global_step += 1
                losses.append(loss)
            except tf.errors.OutOfRangeError:
                break
        print("train batch loss this epoch: %f" %(np.mean(losses)))

    def eval(self):
        self.sess.run(self.validation_init_op)
        total_loss = 0
        total_count = 0
        max_dev = []
        while True:
            try:
                gt, pred, loss = self.sess.run([self.result, self.model.pred, self.model.loss])
                total_loss += loss
                total_count += gt.shape[0]*gt.shape[1]
                max_d = np.amax(np.linalg.norm(gt-pred, axis=2), axis=1)
                max_dev.extend(max_d.tolist())
            except tf.errors.OutOfRangeError:
                break
        print("eval average node L2 loss: %f" % (total_loss/total_count))
        print("eval max deviation: %f" %(np.mean(max_dev)))

    def train(self):
        for i in range(self.num_epoch):
            self.train_epoch()
            self.eval()
            self.model.save(self.sess, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=['linear', 'fc_concat', 'fc_add', 'conv_add', 'LSTM'],
                                      help="model type")
    args = parser.parse_args()
    train_dataset = 'datasets/neuralsim_train_s9ka10.tfrecords'
    test_dataset = 'datasets/neuralsim_test_s1ka10.tfrecords'

    trainner = Trainner(args.model_type, train_dataset, test_dataset, 35, 640)
    trainner.train()

