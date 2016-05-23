import numpy as np
import os
import Image

np.random.seed(1337)  # for reproducibility

#mnist_split_to_test_and_train.py

import random
from keras.datasets import mnist

def do_split():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	os.mkdir('train')
	os.mkdir('test')
	
	np.savetxt('labels_train.csv', y_train, header='label')
	np.savetxt('labels_test.csv', y_test, header='label')
	
	for i in xrange(X_train.shape[0]):
		im = Image.fromarray(np.uint8(X_train[i]))
		im.save('train'+str(i)+'.png')
	
	for i in xrange(X_test.shape[0]):
		im = Image.fromarray(np.uint8(X_test[i]))
		im.save('test'+str(i)+'.png')	