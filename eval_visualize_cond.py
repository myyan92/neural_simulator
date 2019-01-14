import tensorflow as tf
from model_wrapper import Model
from build_dataset import data_parser
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, model_type, eval_datasets, eval_params, eval_snapshot):

        # create TensorFlow Dataset objects
        val_datasets = [tf.data.TFRecordDataset(vd) for vd in eval_datasets]
        val_datasets = [val_data.map(data_parser) for val_data in val_datasets]
        val_datasets = [val_data.map(lambda t0, t1, t2: (t0, t1, t2, tf.constant(val_params)) )
                          for val_data, val_params in zip(val_datasets, eval_params)]
        val_choice_dataset = tf.data.Dataset.range(len(eval_datasets)).repeat()
        val_data = tf.contrib.data.choose_from_datasets(val_datasets, val_choice_dataset)
        val_data = val_data.batch(len(eval_datasets))
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                                   val_data.output_shapes)
        self.start, self.action, self.result, self.param = iterator.get_next() # self.next_gradient
        # create two initialization ops to switch between the datasets
        self.eval_init_op = iterator.make_initializer(val_data)

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.model = Model(model_type)  #load pretrained weights
        self.model.build(input=self.start, action=self.action, param=self.param)
        self.model.setup_optimizer(0, self.result)
        self.sess.run(tf.global_variables_initializer())
        self.model.load(self.sess, eval_snapshot)

    def eval(self):
        self.sess.run(self.eval_init_op)
        total_loss = 0
        total_count = 0
        while True:
            try:
                start, action, result, pred = self.sess.run([self.start, self.action, self.result,
                                                             self.model.pred])
                loss = np.sum(np.square(result-pred))/2.0
                total_loss += loss
                total_count += result.shape[0] * result.shape[1]
                #np.savetxt('../pred/%04d_knots.txt'%(total_count), pred[0].transpose())
                for i in range(start.shape[0]):
                    plt.figure()
                    plt.plot(start[i,:,0], start[i,:,1], label='start')
                    plt.plot(result[i,:,0], result[i,:,1], label='GT')
                    plt.plot(pred[i,:,0], pred[i,:,1], label='pred')
                    idx = np.where(action[i] != 0)
                    if len(idx[0])>0:
                        node = idx[0][0]
                        print(node)
                        plt.arrow(start[i,node,0], start[i,node,1],
                                  result[i,node,0]-start[i,node,0],result[i,node,1]-start[i,node,1])
                    plt.legend()
                    plt.axis("equal")
                    plt.show()
                    #plt.savefig("vis_%03d.png"%(total_count // result.shape[1]))
                    plt.close()
            except tf.errors.OutOfRangeError:
                break
        print("eval average node L2 loss: %f" % (total_loss/total_count))

if __name__ == '__main__':
    eval_datasets= ['datasets/neuralsim_test_s1ka10_b1.tfrecords',
                    'datasets/neuralsim_test_s1ka10_b100.tfrecords',
                    'datasets/neuralsim_test_s1ka10_b10000.tfrecords']
    eval_params = [np.array([np.log(1.0)], dtype=np.float32),
                   np.array([np.log(100.0)], dtype=np.float32),
                   np.array([np.log(10000.0)], dtype=np.float32)]
    vis = Visualizer('cond_LSTM', eval_datasets, eval_params, 'model-11')
    vis.eval()

