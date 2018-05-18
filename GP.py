import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import random

#mutation, crossover, and reproduction operations are 0.19, 0.80, and 0.01


def crossover(v, v0, w, h):
	ind1 = random.randint(0, w * h)
	ind2 = random.randint(0, w * h)
	ind11 = min(ind1, ind2)
	ind22 = max(ind1, ind2)
	tmp = v[ind11 : ind22]
	v[ind11 : ind22] = v0[ind11 : ind22]
	v0[ind11 : ind22] = tmp
	return v

def mutation(v, w, h):
	for i in range(w * h):
    	    if random.random() > 0.7:
                #print np.random.normal(0, 0.5)
	        v[i] = v[i] + np.random.normal(0, 0.5)
                    #if v[i] == 1:
			#	v[i] = 0
			#else:
			#	v[i] = 1
	return v



def GP(img, img0, seed):
    w, h = img.shape
    vimg = np.reshape(img, (w * h, 1))
    vimg0 = np.reshape(img0, (w * h, 1))
    # seed = random.random()
    # print "seed:", seed
    if seed < 0.8:
    	next_vimg = crossover(vimg, vimg0, w, h)

    elif (seed >= 0.8) and (seed < 0.99):
    	next_vimg = mutation(vimg, w, h)

    else:
    	next_vimg = vimg

    next_img = np.reshape(next_vimg, (w, h))

    return next_img



#w = 10
#h = 10
#a = np.random.randint(0, 2, size = [w, h])
#aa = np.random.randint(0, 2, size = [w, h])
#a = float(a)
#aa = float(aa)
#b = GP(a, aa)
#print b
#plt.imshow(b)
#plt.axis('off')
#plt.show()
