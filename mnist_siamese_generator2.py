'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_siamese_generator.py

'''
from __future__ import absolute_import
from __future__ import print_function


import random
import Image
import os

from multiprocessing import Pool

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K

import numpy as np
np.random.seed(1337)  # for reproducibility

def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return shape1


def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model
    
def create_pairs(x, digit_indices):
	'''Positive and negative pair creation.
	Alternates between positive and negative pairs.
	'''
	pairs = []
	labels = []
	n = min([len(digit_indices[d]) for d in range(10)]) - 1
	for d in range(10):
		for i in range(n):
			z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
			pairs += [[x[z1], x[z2]]]
			inc = random.randrange(1, 10)
			dn = (d + inc) % 10
			z1, z2 = digit_indices[d][i], digit_indices[dn][i]
			pairs += [[x[z1], x[z2]]]
			labels += [1, 0]
	return np.array(pairs), np.array(labels)
	
def create_pairs2(x, digit_indices):
	'''Positive and negative pair creation.
	Alternates between positive and negative pairs.
	'''

	n_pairs = 2*len(np.concatenate(digit_indices))
	n_values = len(digit_indices)
	pairs = np.zeros((n_pairs, 2) + x.shape[1:])
	labels = np.zeros(n_pairs)
	
	q = 0
	while q < n_pairs/2:
		i = np.random.randint(n_values)
		if len(digit_indices[i]) <= 1:
			continue
		j,k = np.random.choice(digit_indices[i], replace=False, size=2)
		pairs[2*q, 0] = x[j]
		pairs[2*q, 1] = x[k]
		labels[2*q] = 1
		q += 1

	q = 0
	useable_indices = [i for i in range(n_values) if len(digit_indices[i]) > 1]
	while q < n_pairs/2:
		i, i2 = np.random.choice(useable_indices, replace=False, size=2)
		j = np.random.choice(digit_indices[i], replace=False, size=1)
		k = np.random.choice(digit_indices[i2], replace=False, size=1)
		pairs[2*q+1, 0] = x[j]
		pairs[2*q+1, 1] = x[k]
		q += 1		

	return pairs, labels

'''
def create_all_pairs(x, labels):
	n = x.shape[0]
	all_pairs = np.zeros((n*(n-1)/2,2) + x.shape)
	for i in xrange(n):
		for j in xrange(i+1, n):
			all_pairs(n*i+j*i
'''		

'''
def create_base_network(input_dim):
	#Base network to be shared (eq. to feature extraction).
	
	seq = Sequential()
	seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))
	return seq
'''	


def compute_accuracy(predictions, labels):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return labels[predictions.ravel() < 0.5].mean()


def create_base_network(input_dim):
	# input image dimensions
	img_colours, img_rows, img_cols = input_dim

	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	nb_pool = 2
	# convolution kernel size
	nb_conv = 3
	model = Sequential()

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
							border_mode='valid',
							input_shape=(img_colours, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	#model.add(Dropout(0.1)) #0.25 #too much dropout and loss -> nan

	model.add(Flatten())
	
	model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
	#model.add(Dropout(0.05))
	model.add(Dense(32, activation='relu'))

	
	'''
	model.add(Dense(32)) #128
	model.add(Activation('relu'))
	model.add(Dropout(0.1)) #0.5
	model.add(Dense(10, activation='tanh')) #128
	#model.add(Dense(10))
	#model.add(Activation('softmax'))
	'''
	
	return model


def preprocess(X):
	#this preprocessor crops one pixel along each of the sides of the images
	return X[:, :, 1:-1, 1:-1] / 255.0	
	#return X/ 255.0	


def filereader(fname):
	x = np.array(Image.open(fname))
	
	if len(x.shape) == 2:
		#add an additional colour dimension if the only dimensions are width and height
		return preprocess( x.reshape((1, 1) + x.shape) )
	if len(x.shape) == 3:
		return preprocess( x.reshape((1) + x.shape) )	
		
	#return np.array(Image.open('train/train'+str(n)+'.png'))[1:-1, 1:-1]
	

def myGenerator(y_train, chunk_size, batch_size):
	poss_values = np.unique(y_train).astype(int)
	
	#read and preprocess first file to figure out the image dimensions
	sample_file = filereader('train/train'+str(0)+'.png')
	new_img_colours, new_img_rows, new_img_cols = sample_file.shape
	
	pool = Pool(processes=16)
	
	while 1:
		for i in xrange(y_train.shape[0]/chunk_size):
			X_train = pool.map(filereader, ['train/train'+str(i*chunk_size+i2)+'.png'  for i2 in xrange(chunk_size)])
			X_train = np.array(X_train).reshape((chunk_size, new_img_colours, new_img_rows, new_img_cols)) #.astype('float32')
						
			for j in xrange(int(chunk_size/batch_size)): 	
				digit_indices = [np.where(y_train[i*chunk_size+j*batch_size:i*chunk_size+(j+1)*batch_size] == k)[0] for k in poss_values]
				tr_pairs, tr_y = create_pairs2(X_train[j*batch_size:(j+1)*batch_size], digit_indices)
				yield [tr_pairs[:, 0], tr_pairs[:, 1]], tr_y



def do_split():
	if os.path.isdir('train') and os.path.isdir('test'):
		return
	
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


def fit_model():	
	#unzip the mnist data into train and test directories and create labels_train.csv and labels_test.csv
	do_split()
	
	#load all the labels for the train and test sets
	y_train = np.loadtxt('labels_train.csv')
	y_test = np.loadtxt('labels_test.csv')

	# input image dimensions
	img_rows, img_cols = 28, 28

	X_test = np.zeros((y_test.shape[0], 1, img_rows, img_cols))
	for j in xrange(y_test.shape[0]):
		X_test[j] = np.array(Image.open('test/test'+str(j)+'.png'))

	X_test = X_test.reshape(-1, 1, img_rows, img_cols)	


	nb_epoch = 12
	batch_size = 32
	chunk_size = y_train.shape[0]/4

	# create pairs of images in the test set
	digit_indices = [np.where(y_test == i)[0] for i in range(10)]
	#te_pairs, te_y = create_pairs(X_test, digit_indices)
	te_pairs, te_y = create_pairs(preprocess(X_test), digit_indices)

	new_img_colours, new_img_rows, new_img_cols = te_pairs[0].shape[-3:]

	# network definition
	#base_network = create_base_network(input_dim)
	base_network = create_base_network((new_img_colours, new_img_rows, new_img_cols))

	input_a = Input(shape=(new_img_colours, new_img_rows, new_img_cols,))
	input_b = Input(shape=(new_img_colours, new_img_rows, new_img_cols,))

	# because we re-use the same instance 'base_network',
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)

	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	model = Model(input=[input_a, input_b], output=distance)

	# train
	rms = RMSprop()
	model.compile(loss=contrastive_loss, optimizer=rms)

	'''
	model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
			  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
			  batch_size=128,
			  nb_epoch=nb_epoch)
	'''		  
		  
		  
	model.fit_generator(myGenerator(y_train, chunk_size, batch_size), samples_per_epoch = y_train.shape[0], nb_epoch = nb_epoch, verbose=2,callbacks=[], validation_data=None, class_weight=None) # show_accuracy=True, nb_worker=1 
		  

	# compute final accuracy on training and test sets
	#pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
	#tr_acc = compute_accuracy(pred, tr_y)
	#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

	pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
	te_acc = compute_accuracy(pred, te_y)
	print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))