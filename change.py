import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import random

def addnoisy(img, var):
	w, h = img.shape
	noisy = np.random.normal(0, var, (w, h))
        print noisy
	return img + noisy

def rotate90(img):
	w, h = img.shape
	img90 = np.random.normal(0, 0, (w, h))
	for i in range(w):
		for j in range(h):
			img90[h - 1 - j, i] = img[i, j]

	return img90

def rotate_90(img):
	w, h = img.shape
	img_90 = np.random.normal(0, 0, (w, h))
	for i in range(w):
		for j in range(h):
			img_90[j, w - 1 - i] = img[i, j]

	return img_90


#w = 9
#h = 9
#a = np.random.randint(0, 2, size = [w, h])
#aa = np.random.randint(0, 2, size = [w, h])
#b = rotate90(a)
#c = rotate_90(b)
#print a
#print b
#print c
#plt.imshow(b)
#plt.axis('off')
#plt.show()
