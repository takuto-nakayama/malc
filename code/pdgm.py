#  importing modules
from ripser import ripser
import numpy as np
import argparse, os


#  main process
if __name__ == '__main__':
	## parsing arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('path', type=str)
	parser.add_argument('--metric', type=str, default='cosine')
	parser.add_argument('--save_path', type=str, default='../output/pdgm/')
	args = parser.parse_args()

	## defining variables
	path = args.path
	metric = args.metric
	save_path = args.save_path

	## computing persistent homology
	### if path is a file
	embedding = np.load(path)
	dgms = ripser(embedding, metric='cosine')['dgms']
	### saving results
	np.save(f'{save_path}/h0/{p}-h0.npy', dgms[0])
	np.save(f'{save_path}/h1/{p}-h1.npy', dgms[1])

	### if path is a directory
	if os.path.isdir(path):
		paths = os.listdir(path)
	for p in paths:
		embedding = np.load(p)
		dgms = ripser(embedding, metric='cosine')['dgms']
		### saving results
		np.save(f'{save_path}/h0/{p}-h0.npy', dgms[0])
		np.save(f'{save_path}/h1/{p}-h1.npy', dgms[1])