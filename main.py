import numpy as np
import sugartensor as tf

import sys, os
import glob
import time

import argparse

# import GP
from GP import *
from change import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2' # GPU ID

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='ENGG5189 Project Run script')
parser.add_argument('-p','--phase', type=str, default='train', help='Different phase of network, options: train, test', required=False)
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Mnist train data batch size, default: 64', required=False)
parser.add_argument('-max_ep', '--max_epoch', type=int, default=5, help='Epoch Loop number, default: 5', required=False)
parser.add_argument('-dir','--save_dir', type=str, default='./asset/train/gp_model2', help='Trained model save dir', required=False)
parser.add_argument('-v','--var', type=float, default=0.2, help='nosiy variant, default: 0.2', required=False)
parser.add_argument('-an', '--addnoise', action='store_true', help='add noise flag', required=False)
parser.add_argument('-rt', '--rotate', action='store_true', help='rotate flag', required=False)
opt = parser.parse_args()

batch_size = opt.batch_size
max_ep = opt.max_epoch
save_dir = opt.save_dir
var = opt.var

addnoise = opt.addnoise
rotate = opt.rotate

# MNIST data
# images => Mnist.train.image/Mnist.valid.image/Mnist.test.image
# labels => Mnist.train.label/Mnist.valid.label/Mnist.test.label
Mnist = tf.sg_data.Mnist(batch_size=batch_size) # shuffle one 

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

	def train(self): # train baseline model
		input_ph = tf.placeholder(shape=[batch_size, 28, 28, 1], dtype=tf.float32)
		label_ph = tf.placeholder(shape=[batch_size,], dtype=tf.int32)

		predict = self.forward(input_ph)

		loss_tensor = tf.reduce_mean(predict.sg_ce(target=label_ph))

		# use to update network parameters
		optim = tf.sg_optim(loss_tensor, optim='Adam', lr=1e-3)

		# use saver to save a new model 
		saver = tf.train.Saver()

		sess = tf.Session()
		with tf.sg_queue_context(sess):
			# inital 
			tf.sg_init(sess)

		# validation
		acc = (predict.sg_reuse(input=Mnist.valid.image).sg_softmax()
					.sg_accuracy(target=Mnist.valid.label, name='validation'))

		tf.sg_train(loss=loss, eval_metric=[acc], max_ep=max_ep, save_dir=save_dir, ep_size=Mnist.train.num_batch, log_interval=10)

	# test using batch size = 1
	def test(self):
		print 'Testing model {}: addnoise={}, rotate={}, var={}'.format(save_dir, addnoise, rotate, var)
		input_ph = tf.placeholder(shape=[batch_size, 28, 28, 1], dtype=tf.float32)
		label_ph = tf.placeholder(shape=[batch_size,], dtype=tf.int32)

		predict = self.forward(input_ph)

		acc = (predict.sg_softmax()
				.sg_accuracy(target=label_ph, name='test'))

		sess = tf.Session()
		with tf.sg_queue_context(sess):
			tf.sg_init(sess)			

			saver=tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint(save_dir))

			total_accuracy = 0
			for i in range(Mnist.test.num_batch):
				[image_array, label_array] = sess.run([Mnist.test.image, Mnist.test.label])	

				if addnoise:
					image_array[0, :, :, 0] = addnoisy(image_array[0, :, :, 0], var)
				if rotate:
					image_array[0, :, :, 0] = rotate_90(image_array[0, :, :, 0])

				acc_value = sess.run([acc], feed_dict={input_ph: image_array, label_ph: label_array})[0]
				total_accuracy += np.sum(acc_value)

			print 'Evaluation accuracy: {}'.format(float(total_accuracy)/(Mnist.test.num_batch*batch_size))

		# close session
		sess.close()

#############################################################################################################################################

	# train with GP using batch size = 1
	def train_with_GP(self):
		input_ph = tf.placeholder(shape=[batch_size, 28, 28, 1], dtype=tf.float32)
		label_ph = tf.placeholder(shape=[batch_size,], dtype=tf.int32)

		predict = self.forward(input_ph)

		loss_tensor = tf.reduce_mean(predict.sg_ce(target=label_ph))

		# use to update network parameters
		optim = tf.sg_optim(loss_tensor, optim='Adam', lr=1e-3)

		# use saver to save a new model 
		saver = tf.train.Saver()

		sess = tf.Session()
		with tf.sg_queue_context(sess):
			# inital 
			tf.sg_init(sess)

			# train by GP guilding
			for e in range(max_ep):
				previous_loss = None
				for i in range(Mnist.train.num_batch):
					[image_array, label_array] = sess.run([Mnist.train.image, Mnist.train.label])	

					if (e == 0 or e == 1): # first and second epoch train no noisy image 
						loss = sess.run([loss_tensor, optim], feed_dict = {input_ph: image_array, label_ph: label_array})[0]
						print 'Baseline loss = ', loss
					elif (e == 2): # third epoch train with gp image and original image
						gpIn1 = np.squeeze(image_array)
						gpIn2 = np.zeros((28, 28))
						image_gp = GP(gpIn1, gpIn2, seed=0.8)
						image_gp2 = image_gp[np.newaxis, ...]
						image_gp2 = image_gp2[..., np.newaxis]
						loss = sess.run([loss_tensor, optim], feed_dict={input_ph: image_array, label_ph: label_array})[0]
						print 'GP without nosiy loss = ', loss							
						loss = sess.run([loss_tensor, optim], feed_dict={input_ph: image_gp2, label_ph: label_array})[0]
						print 'GP loss = ', loss					
					else: # other epoch train with gp evolution 
						gpIn1 = np.squeeze(image_array)
						gpIn2 = np.zeros((28, 28))
						image_gp = GP(gpIn1, gpIn2, seed=random.random())
						image_gp2 = image_gp[np.newaxis, ...]
						image_gp2 = image_gp2[..., np.newaxis]		
						loss = sess.run([loss_tensor, optim], feed_dict={input_ph: image_array, label_ph: label_array})[0]
						print 'GP without nosiy loss = ', loss											
						loss = sess.run([loss_tensor, optim], feed_dict={input_ph: image_gp2, label_ph: label_array})[0]
						print 'GP loss = ', loss
						if loss < previous_loss:
							for i in range(5):
								loss = sess.run([loss_tensor, optim], feed_dict={input_ph: image_gp2, label_ph: label_array})[0]
								gpIn1 = image_gp2
								image_gp2[0, :, :, 0] = GP(gpIn1[0,:,:,0], gpIn2, seed=random.random())
								print 'GP EV loss = ', loss

				previous_loss = loss
				saver.save(sess, os.path.join(save_dir, 'gp_model'), global_step=e)
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
	elif phase.lower() == 'train_gp':
		# test only one batch 
		model.train_with_GP()
	else:
		pass