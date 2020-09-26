import tensorflow as tf
from dataset_io import data_parser
import numpy as np
import functools
import matplotlib.pyplot as plt
import pdb

@gin.configurable
class Visualizer():
    def __init__(self, model, eval_dataset, eval_snapshot):

        # create TensorFlow Dataset objects
        val_data = tf.data.TFRecordDataset(eval_dataset)
        parser_noaug = functools.partial(data_parser, dim=model.dim, augment=False)
        val_data = val_data.map(parser_noaug)
        val_data = val_data.batch(1)
        # create TensorFlow Iterator object
        iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                                   val_data.output_shapes)
        self.start, self.action, self.result = iterator.get_next()
        # create two initialization ops to switch between the datasets
        self.eval_init_op = iterator.make_initializer(val_data)

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.model = model
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
                start, action, result, pred = self.sess.run([self.start, self.action, self.result,
                                                             self.model.pred])
                max_dev = np.amax(np.linalg.norm(result-pred, axis=2), axis=1)
                loss = np.sum(np.square(result-pred))/2.0
                total_loss += loss
                total_count += result.shape[0] * result.shape[1]
                plt.clf()
                fig = plt.gcf()
                fig.set_size_inches(8,8)
                plt.plot(start[0,:,0], start[0,:,1], c='tab:blue', linewidth=1, label='start')
                plt.plot(result[0,:,0], result[0,:,1], c='tab:green', linewidth=0.5, label='GT')
                plt.plot(pred[0,:,0], pred[0,:,1], c='tab:red', linewidth=0.5, label='pred')
                idx = np.where(action != 0)
                if len(idx[0])>0:
                    node = idx[1][0]
                    plt.arrow(start[0,node,0], start[0,node,1], action[0,node,0], action[0,node,1])
                plt.legend()
                plt.axis("equal")
                axes = plt.gca()
                plt.show()
                #plt.savefig("vis_%03d.png"%(total_count // result.shape[1]),bbox_inches='tight')
                #plt.close()
                #print("saved")
            except tf.errors.OutOfRangeError:
                break
        print("eval average node L2 loss: %f" % (total_loss/total_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_config', default='', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)

    vis = Visualizer()
    vis.eval()

