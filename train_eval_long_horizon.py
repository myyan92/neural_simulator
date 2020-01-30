import tensorflow as tf
import datetime
import sys
import os
from model_long_horizon import Model
from dataset_io_long_horizon import data_parser, batch_map_fn
import numpy as np
import argparse
import pdb
import matplotlib.pyplot as plt

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

class Trainner():
    def __init__(self, train_dataset, eval_dataset, num_epoch, batch_size):

        # create TensorFlow Dataset objects
        tr_data = tf.data.TFRecordDataset(train_dataset)
        tr_data = tr_data.map(data_parser) # .filter(lambda a,b,c: tf.norm(a)>1e-3)
        tr_data = tr_data.shuffle(10000).padded_batch(batch_size, ([10,64,3],[10,64,3],[10,64,3],[None,3],[None,3])).map(batch_map_fn)
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser).padded_batch(batch_size, ([10,64,3],[10,64,3],[10,64,3],[None,3],[None,3])).map(batch_map_fn)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                   tr_data.output_shapes)
        self.start, self.action, self.result, self.gather_index_over, self.gather_index_under = iterator.get_next()  # self.next_gradient
        # create two initialization ops to switch between the datasets
        self.training_init_op = iterator.make_initializer(tr_data)
        self.validation_init_op = iterator.make_initializer(val_data)

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.num_epoch = num_epoch
        self.model = Model()
        self.model.build(steps=10,input=self.start, action=self.action)
        self.model.setup_optimizer(0.001, self.result, self.gather_index_over, self.gather_index_under)
        self.global_step = 0
        self.trial_name = 'GRU_attention_topo_reg_2'
        self.train_writer = tf.summary.FileWriter('tboard/train_{}/'.format(self.trial_name), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.f = open('./train_{}_log.txt'.format(self.trial_name), 'w')

    def train_epoch(self):

        self.sess.run(self.training_init_op)
        losses = []
        grads = []
        while True:
            try:
#                start_val, action_val, result_val, index_over_val, index_under_val = self.sess.run([self.start, self.action, self.result, self.gather_index_over, self.gather_index_under])
#                if np.prod(index_over_val.shape)>0 and np.amax(index_over_val) >=64:
#                    pdb.set_trace()
                summary, _, loss = self.sess.run([self.model.merged_summary, self.model.optimizer, self.model.loss])
                self.train_writer.add_summary(summary, self.global_step)
                self.global_step += 1
                losses.append(loss)
            except tf.errors.OutOfRangeError:
                break
        ori = sys.stdout
        sys.stdout = Tee(sys.stdout, self.f)
        print("\rEPOCH",self.cur_epoch, datetime.datetime.now()) 
        print("    train batch loss this epoch: {}".format(np.mean(losses)))
        sys.stdout = ori

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
                total_count += gt.shape[0]*gt.shape[1]*gt.shape[2]
                max_d = np.amax(np.linalg.norm(gt-pred, axis=3), axis=2)
                max_dev.extend(max_d.tolist())
            except tf.errors.OutOfRangeError:
                break
        ori = sys.stdout
        sys.stdout = Tee(sys.stdout, self.f)
        print("    eval average node L2 loss: %f" % (total_loss/total_count))
        print("    eval max deviation: %f" %(np.mean(max_dev)))
        sys.stdout = ori
        #print(np.histogram(max_dev))

    def train(self):
        for i in range(self.num_epoch):
            self.cur_epoch = i
            self.train_epoch()
            self.model.save(self.sess, './models_{}/model'.format(self.trial_name), i)
            self.eval()

if __name__ == '__main__':
    train_dataset = 'datasets/neuralsim_train_simseq3d_long_horizon_reg.tfrecords'
    test_dataset = 'datasets/neuralsim_test_simseq3d_long_horizon_reg.tfrecords'

    trainner = Trainner(train_dataset, test_dataset, 10000, 128)
    trainner.train()
#    pretrain_path = './models_LSTM_attention_topo/model-290'
#    trainner.model.load(trainner.sess, pretrain_path)
#    trainner.eval()
