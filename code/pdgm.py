#  importing modules
from ripser import ripser
import numpy as np
import argparse, os, re


#  main process
if __name__ == '__main__':
	## parsing arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('path', type=str)
	parser.add_argument('id', type=str)
	parser.add_argument('--sample_size', type=int, default=10000)
	parser.add_argument('--metric', type=str, default='cosine')
	parser.add_argument('--save_path', type=str, default='../output/pdgm/')
	args = parser.parse_args()

	## defining variables
	path = args.path
	id = args.id
	sample_size = args.sample_size
	metric = args.metric
	save_path = args.save_path

	## computing persistent homology
	embedding = np.load(path)[:sample_size]
	dgms = ripser(embedding, metric='cosine')['dgms']
	### saving results
	np.save(f'{save_path}/h0/{id}-h0.npy', dgms[0])
	np.save(f'{save_path}/h1/{id}-h1.npy', dgms[1])