'''
Train a siamese convolution neural network on pairs of digits from the MNIST dataset.

This code was adapted by Small Yellow Duck (https://github.com/small-yellow-duck/) from 
https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py

The similarity between two images is calculated as per Hadsell-et-al.'06 [1] by 
computing the Euclidean distance on the output of the shared network and by 
optimizing the contrastive loss (see paper for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_siamese_cnn.py

'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K


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



def compute_accuracy(predictions, labels):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return labels[predictions.ravel() < 0.5].mean()


'''
def create_base_network_dense(input_dim):
	#Base network to be shared (eq. to feature extraction).
	
	seq = Sequential()
	seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))
	return seq
'''	


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
	#model.add(Dropout(0.05)) #too much dropout and loss -> nan
	model.add(Dense(32, activation='relu'))

	

	return model


		

	
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)

# input image dimensions
img_rows, img_cols = 28, 28
X_train = X_train.reshape(60000, 1, img_rows, img_cols)
X_test = X_test.reshape(10000, 1, img_rows, img_cols)
	
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#input_dim = 784
input_dim = (1, img_rows, img_cols)
nb_epoch = 12

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

# network definition
#base_network = create_base_network(input_dim)
base_network = create_base_network(input_dim)

input_a = Input(shape=(1, img_rows, img_cols,))
input_b = Input(shape=(1, img_rows, img_cols,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
		  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
		  batch_size=128,
		  nb_epoch=nb_epoch)
#model.fit_generator(myGenerator(), samples_per_epoch = 60000, nb_epoch = 2, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
		  

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))