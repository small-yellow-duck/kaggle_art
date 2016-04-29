'''
make_overlaps() generates an image of a matrix which indicates how much the 
styles of different artists overlap. 
the style information was extracted from wikipedia

author: Small Yellow Duck
https://github.com/small-yellow-duck/kaggle_art
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import networkx as nx
from networkx.utils import cuthill_mckee_ordering




def sort2d(x):
	G = nx.DiGraph(x).to_undirected()
	rcm = list(cuthill_mckee_ordering(G))
	return x[:,rcm][rcm,:], rcm
		
		
#overlaps, artists = np.loadtxt('overlaps.csv')	
#overlaps, artists = generate_overlaps(train_info)
def generate_overlaps(all_info):	
	b = all_info
	
	b['short_style'] = b['style'].apply(lambda x : str(x).lower().replace('art ', '').replace(' art', ''))
	b['short_style'] = b['short_style'].apply(lambda x : x.split(' ')[-1])
	b['short_style'] = b['short_style'].apply(lambda x : x.replace('(', '').replace(')', '').lower())
	
	q = b.groupby('short_style').artist.nunique().reset_index()
	np.mean(q.artist > 1)
	d = b.groupby(['artist', 'short_style']).size().reset_index()
	
	e = pd.pivot_table(d, index='artist', columns='short_style', values=0, fill_value=0)
	f = 1.0*e.iloc[:, 1:].div(e.iloc[:, 1:].sum(axis=1), axis=0)
	
	n = f.shape[0]
	overlaps = np.ones((n,n))
	for i in xrange(1, n):
		for j in xrange(i+1, n):
			overlaps[i,j] = np.sum(f.iloc[i, :].values * f.iloc[j, :].values)
			overlaps[j,i] = overlaps[i,j]

	return overlaps, f.index


def make_overlaps():
	train_info = pd.read_csv('train_info.csv')	
	overlaps, artists = generate_overlaps(train_info)
	overlaps_sorted = sort2d(overlaps)	
	plt.close('all')
	fig, ax = plt.subplots()
	im = ax.imshow(overlaps_sorted[0][:, :], interpolation='nearest')
	#ax.set_title('artist overlap by style')
	ax.xaxis.tick_top()
	ax.set_xlabel('artist')
	ax.set_ylabel('artist')
	cbar = plt.colorbar(im)
	cbar.set_label('artist style overlap', rotation=90)
 