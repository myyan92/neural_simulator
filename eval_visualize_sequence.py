import tensorflow as tf
from model_wrapper import Model
from build_dataset import data_parser
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, model_type, eval_dataset, eval_snapshot, plot_idx):

        self.plot_idx = plot_idx
        # create TensorFlow Dataset objects
        val_data = tf.data.TFRecordDataset(eval_dataset)
        val_data = val_data.map(data_parser)
        val_data = val_data.batch(1)
        # create TensorFlow Iterator object
        self.iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                                   val_data.output_shapes)
        # create two initialization ops to switch between the datasets
        self.eval_init_op = self.iterator.make_initializer(val_data)
        self.start, self.action, self.result = self.iterator.get_next() # self.next_gradient

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.model = Model(model_type)  #load pretrained weights
        self.model.build(input=self.start, action=self.action)
        self.model.setup_optimizer(0, self.result)
        self.sess.run(tf.global_variables_initializer())
        self.model.load(self.sess, eval_snapshot)

    def eval(self):
        self.sess.run(self.eval_init_op)
        total_loss = 0
        total_count = 0
        while True:
            try:
                if total_count == 0:
                    start, action, result, pred = self.sess.run([self.start, self.action,
                        self.result, self.model.pred])
                    last_end = pred
                else:
                    start, action, result, pred = self.sess.run([self.start, self.action,
                        self.result, self.model.pred],
                        feed_dict={self.start:last_end})

                plt.clf()
                fig = plt.gcf()
                fig.set_size_inches(8,8)
                if total_count == 0:
                    plt.plot(start[0,:,0], start[0,:,1], c='#4e79a788', linewidth=1, label='gt_start')
                else:
                    plt.plot(last_result[0,:,0], last_result[0,:,1], c='#4e79a788', linewidth=1, label='gt_start')

                plt.plot(result[0,:,0], result[0,:,1], c='#f28e2b88', linewidth=0.5, label='gt_end')
                plt.scatter(last_end[0,:,0], last_end[0,:,1], marker='.', c='tab:blue', s=5, label='last_pred')
                plt.scatter(pred[0,:,0], pred[0,:,1], marker='.', c='tab:orange', s=5, label='pred')
                idx = np.where(action != 0)
                if len(idx[0])>0:
                    node = idx[1][0]
                    if total_count > 0:
                        plt.arrow(last_result[0,node,0], last_result[0,node,1], action[idx][0], action[idx][1],
                                color='#59a14f88',length_includes_head=True)
                        plt.arrow(last_end[0,node,0], last_end[0,node,1], action[idx][0], action[idx][1],
                                color='tab:green',length_includes_head=True)
                    else:
                        plt.arrow(start[0,node,0], start[0,node,1], action[idx][0], action[idx][1],
                                color='#59a14f88',length_includes_head=True)
                                    # result[0,node,0]-start[0,node,0],result[0,node,1]-start[0,node,1])
                plt.legend()
                plt.axis("equal")                
                axes = plt.gca()
                axes.set_xlim([-0.6,0.6])
                axes.set_ylim([-0.6,0.6])
                #plt.show()
                plt.savefig("vis_{:03d}_{:04d}.png".format(self.plot_idx, total_count),bbox_inches='tight')
                print(total_count, action[idx],node)
                last_end = pred
                last_result = result
                total_count+=1
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    for i in range(1,20):
        vis = Visualizer('LSTM', 'datasets/neuralsim_test_{}.tfrecords'.format(9000+i*10), 'models_best/model-best', i)
        vis.eval()
        tf.reset_default_graph()
