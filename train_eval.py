import tensorflow as tf
from model_wrapper import Model
from dataset_io_new import data_parser  # for 3d state and 5d action.
import numpy as np
import argparse
import gin
import pdb
import matplotlib.pyplot as plt

class Trainner():
    def __init__(self, model_type, train_dataset, eval_dataset, num_epoch, batch_size):

        # create TensorFlow Dataset objects
        tr_data = tf.data.TFRecordDataset(train_dataset)
        tr_data = tr_data.map(data_parser).filter(lambda a,b,c: tf.norm(a)>1e-3)
        tr_data = tr_data.shuffle(10000).batch(batch_size)
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser).batch(batch_size)
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
        self.model = Model(model_type)
        self.model.build(input=self.start, action=self.action)
        self.model.setup_optimizer(0.001, self.result)
        self.global_step = 0
        self.trial_name = 'GRU_attention_topo'
        self.train_writer = tf.summary.FileWriter('tboard/train_{}/'.format(self.trial_name), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self):

        self.sess.run(self.training_init_op)
        losses = []
        grads = []
        while True:
            try:
                summary, _, loss = self.sess.run([self.model.merged_summary, self.model.optimizer, self.model.loss])
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
                input, action, gt, pred, loss = self.sess.run([self.start, self.action, self.result, self.model.pred, self.model.loss]) # changed for debuging
#                for s,a,r,p in zip(input, action, gt, pred):
#                    plt.plot(s[:,0], s[:,1])
#                    plt.plot(r[:,0], r[:,1])
#                    plt.plot(p[:,0], p[:,1])
#                    node=np.where(np.linalg.norm(a,axis=1)>0)[0][0]
#                    plt.plot([s[node,0], s[node,0]+a[node,0]], [s[node,1], s[node,1]+a[node,1]])
#                    plt.show()

                total_loss += loss
                total_count += gt.shape[0]*gt.shape[1]
                max_d = np.amax(np.linalg.norm(gt-pred, axis=2), axis=1)
                max_dev.extend(max_d.tolist())
            except tf.errors.OutOfRangeError:
                break
        print("eval average node L2 loss: %f" % (total_loss/total_count))
        print("eval max deviation: %f" %(np.mean(max_dev)))
        #print(np.histogram(max_dev))

    def train(self):
        for i in range(self.num_epoch):
            self.train_epoch()
            self.model.save(self.sess, './models_{}/model'.format(self.trial_name), i)
            self.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=['linear', 'fc_concat', 'fc_add', 'conv_add', 'LSTM', 'LSTM_attention', 'GRU_attention'],
                                      help="model type")
    args = parser.parse_args()

    train_dataset = 'datasets/neuralsim_train_simseq3d_topochange.tfrecords'
    test_dataset = 'datasets/neuralsim_test_simseq3d_topochange.tfrecords'

    trainner = Trainner(args.model_type, train_dataset, test_dataset, 300, 640)
    trainner.train()
#    pretrain_path = './models_LSTM_attention_topo/model-290'
#    trainner.model.load(trainner.sess, pretrain_path)
#    trainner.eval()
