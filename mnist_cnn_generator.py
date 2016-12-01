'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import Image
from multiprocessing import Pool


def filereader(fname):
	x = np.array(Image.open(fname))
	
	if len(x.shape) == 2:
		#add an additional colour dimension if the only dimensions are width and height
		return preprocess( x.reshape((1, 1) + x.shape) )
	if len(x.shape) == 3:
		return preprocess( x.reshape((1) + x.shape) )	
	
def preprocess(X):
	#this preprocessor crops one pixel along each of the sides of the images
	#return X[:, :, 1:-1, 1:-1] / 255.0	
	return X/ 255.0			


#list(myGenerator(y_train, batch_size, fnames_train))[0]
def myGenerator(y, batch_size, fnames):

	#read and preprocess first file to figure out the image dimensions
	sample_file = filereader(fnames[0])
	new_img_colours, new_img_rows, new_img_cols = sample_file.shape[1:]

	order = np.arange(len(fnames))
	np.random.shuffle(order)
	y = y[order]
	fnames = fnames[order]

	while True:
		for i in xrange(np.ceil(1.0*len(fnames)/batch_size).astype(int)):
			this_batch_size = fnames[i*batch_size :(i+1)*batch_size].shape[0]
			X = np.zeros((this_batch_size, new_img_colours, new_img_rows, new_img_cols)).astype('float32')
			
			for i2 in xrange(this_batch_size):
				X[i2] = filereader(fnames[i*batch_size])
						
			#training set
			if not y is None:	
				yield X, y[i*batch_size:(i+1)*batch_size]
			#test set
			else:
				yield X
					
					



#pred, Y_test = fit()
def fit():
	batch_size = 128
	nb_epoch = 1
	chunk_size = 128*100

	# input image dimensions
	img_rows, img_cols = 28, 28
	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	nb_pool = 2
	# convolution kernel size
	nb_conv = 3

	#load all the labels for the train and test sets
	y_train = np.loadtxt('labels_train.csv')
	y_test = np.loadtxt('labels_test.csv')
	
	fnames_train = np.array(['train/train'+str(i)+'.png' for i in xrange(len(y_train))])
	fnames_test = np.array(['test/test'+str(i)+'.png' for i in xrange(len(y_test))])
	
	nb_classes = len(np.unique(y_train))

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train.astype(int), nb_classes)
	Y_test = np_utils.to_categorical(y_test.astype(int), nb_classes)

	model = Sequential()

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
							border_mode='valid',
							input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='adadelta',
				  metrics=['accuracy'])

	model.fit_generator(myGenerator(Y_train, batch_size, fnames_train), samples_per_epoch = Y_train.shape[0], nb_epoch = nb_epoch, verbose=1,callbacks=[], validation_data=None, class_weight=None, max_q_size=10) # show_accuracy=True, nb_worker=1 
		  

	pred = model.predict_generator(myGenerator(None, batch_size, fnames_test), len(fnames_test), max_q_size=10) # show_accuracy=True, nb_worker=1 

	#score = model.evaluate(X_test, Y_test, verbose=0)
	#print('Test score:', score[0])
	#print('Test accuracy:', score[1])	
	print( 'Test accuracy:', np.mean(np.argmax(pred, axis=1) == np.argmax(Y_test, axis=1)) )
	
	return pred, Y_test	
		  

