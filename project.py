import numpy as np
import sugartensor as tf

import sys, os
import glob
import time

import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '2' # GPU ID

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='ENGG5189 Project Run script')
parser.add_argument('-p','--phase', type=str, default='train', help='Different phase of network, options: train, test', required=False)
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Mnist train data batch size, default: 64', required=False)
parser.add_argument('-max_ep', '--max_epoch', type=int, default=5, help='Epoch Loop number, default: 5', required=False)
parser.add_argument('-dir','--save_dir', type=str, default='./asset/train', help='Trained model save dir', required=False)
opt = parser.parse_args()

batch_size = opt.batch_size
max_ep = opt.max_epoch
save_dir = opt.save_dir

# MNIST data
# images => Mnist.train.image/Mnist.valid.image/Mnist.test.image
# labels => Mnist.train.label/Mnist.valid.label/Mnist.test.label
Mnist = tf.sg_data.Mnist(batch_size=batch_size)

# fully connect network
class Net(object):
	def __init__(self):
		self.scope = 'FC'
		self.network = {}

	def forward(self, inputs):
		# reuse = len([t for t in tf.global_variables() if t.name.startswith(self.scope)]) > 0
		with tf.sg_context(scope=self.scope, act='sigmoid', bn=False):
			self.network['predict'] = (inputs.sg_flatten()
											.sg_dense(dim=20, name='fc1')
											.sg_dense(dim=10, act='linear', name='predict'))

			return self.network['predict']

# Model
class Model(Net):

	def __init__(self):
		Net.__init__(self)

	def train(self):
		predict = self.forward(Mnist.train.image)

		loss = predict.sg_ce(target=Mnist.train.label)

		# validation
		acc = (predict.sg_reuse(input=Mnist.valid.image).sg_softmax()
					.sg_accuracy(target=Mnist.valid.label, name='validation'))

		tf.sg_train(loss=loss, eval_metric=[acc], max_ep=max_ep, save_dir=save_dir, ep_size=Mnist.train.num_batch, log_interval=10)

	def test(self):
		predict = self.forward(Mnist.test.image)

		acc = (predict.sg_softmax()
				.sg_accuracy(target=Mnist.test.label, name='test'))

		sess = tf.Session()
		with tf.sg_queue_context(sess):
			tf.sg_init(sess)

			saver=tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint(save_dir))

			total_accuracy = 0
			for i in range(Mnist.test.num_batch):
				total_accuracy += np.sum(sess.run([acc])[0])

			print 'Evaluation accuracy: {}'.format(float(total_accuracy)/(Mnist.test.num_batch*batch_size))

		# close session
		sess.close()

# main
if __name__ == '__main__':
	model = Model()

	phase = opt.phase

	if phase.lower() == 'train':
		# train
		model.train()
	elif phase.lower() == 'test':
		# test 
		model.test()
	else:
		pass