import numpy as np
import sugartensor as tf

import sys, os
import glob
import time
from change import *
from GP import *
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

		#######GP
		sess = tf.Session()
		with tf.sg_queue_context(sess):
			tf.sg_init(sess)
			trainf = sess.run([Mnist.train.image])[0]
			n, w, h, c = trainf.shape
			print trainf.shape
			np.savetxt('./image.txt', trainf[1, :, :, 0])
			#print trainf[1, :, :, 0]
			#plt.imshow(trainf[1, :, :, 0])
			#plt.axis('off')
			#plt.show()
			#print type(trainf[1, :, :, 0])

			transfer = np.zeros((n, w, h, c))
			for i in range(n):
				candi = random.randint(0, n - 1)
				#print GP(trainf[i, :, :, 0], trainf[candi, :, :, 0])

				#transfer[i, :, :, :] = GP(trainf[i, :, :, :], trainf[candi, :, :, :])
				#print trainsfer[i, :, :, :]
				t = tf.convert_to_tensor(transfer, dtype=tf.float32)
				gp_predict = predict.sg_reuse(input=t)
				#print trainf.shape
		sess.close()                    



 #                #######

	# 	loss = predict.sg_ce(target=Mnist.train.label)

	# 	# validation
	# 	acc = (predict.sg_reuse(input=Mnist.valid.image).sg_softmax()
	# 				.sg_accuracy(target=Mnist.valid.label, name='validation'))

	# 	tf.sg_train(loss=loss, eval_metric=[acc], max_ep=max_ep, save_dir=save_dir, ep_size=Mnist.train.num_batch, log_interval=10)

	def test(self):
		# predict = self.forward(Mnist.test.image)


		# acc = (predict.sg_softmax()
		# 		.sg_accuracy(target=Mnist.test.label, name='test'))

		sess = tf.Session()
		with tf.sg_queue_context(sess):
			tf.sg_init(sess)
			testf = sess.run([Mnist.test.image])[0]
			# print testf.shape
			n, w, h, c = testf.shape
                        tmp0 = np.zeros((n * w, h))
                        tmp02 = np.zeros((n * w, h))
                        tmp05 = np.zeros((n * w, h))
                        tmp08 = np.zeros((n * w, h))
                        tmp90 = np.zeros((n * w, h))
                        tmp_90 = np.zeros((n * w, h))
			for i in range(n):
                            tmp0[i * w : (i + 1) * w, 0 : h] = testf[i, :, :, 0]
                            tmp02[i * w : (i + 1) * w, 0 : h] = addnoisy(testf[i, :, :, 0], 0.2)
                            tmp05[i * w : (i + 1) * w, 0 : h] = addnoisy(testf[i, :, :, 0], 0.5)
                            tmp08[i * w : (i + 1) * w, 0 : h] = addnoisy(testf[i, :, :, 0], 0.8)
                            tmp90[i * w : (i + 1) * w, 0 : h] = rotate90(testf[i, :, :, 0])
                            tmp_90[i * w : (i + 1) * w, 0 : h] = rotate_90(testf[i, :, :, 0])# addnoisy(testf[i, :, :, 0], 0.8)
                            #testf[i, :, :, 0] = addnoisy(testf[i, :, :, 0], 0.0)
			    #testf[i, :, :, 0] = rotate90(testf[i, :, :, 0])
			    #testf[i, :, :, 0] = rotate_90(testf[i, :, :, 0])
		    	    #print testf[i, :, :, 0]
                        np.savetxt('./image0.txt', tmp0)
                        np.savetxt('./image02.txt', tmp02)
                        np.savetxt('./image05.txt', tmp05)
                        np.savetxt('./image08.txt', tmp08)
                        np.savetxt('./image90.txt', tmp90)
                        np.savetxt('./image_90.txt', tmp_90)

			testf_tensor = tf.convert_to_tensor(testf, dtype=tf.float32)
			predict = self.forward(testf_tensor)

			acc = (predict.sg_softmax()
				.sg_accuracy(target=Mnist.test.label, name='test'))            

			saver=tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint(save_dir))

			total_accuracy = 0
			for i in range(Mnist.test.num_batch):
				total_accuracy += np.sum(sess.run([acc])[0])

			print('Evaluation accuracy: {}'.format(float(total_accuracy)/(Mnist.test.num_batch*batch_size)))

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
