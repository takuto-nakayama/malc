#  importing modules
from persim import wasserstein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, os


#  main process
if __name__ == '__main__':
	## parsing arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('path', type=str)
	parser.add_argument('id', type=str)
	parser.add_argument('--save_path', type=str, default='../output')
	args = parser.parse_args()

	## defining variables
	path = args.path
	id = args.id
	save_path = args.save_path
	paths = os.listdir(path)
	was = []

	## computing wasserstein distance
	for i in paths:
		was_i = []
		dgm_i = np.load(f'{path}/{i}')
		for j in paths:
			if i != j:
				dgm_j = np.load(f'{path}/{j}')
				was_i.append(wasserstein(dgm_i, dgm_j))
			else:
				was_i.append(0)
		was.append(was_i)

	## saving wasserstein distance matrix
	df = pd.DataFrame(was)
	df.to_csv(f'{save_path}/wasserstein-{id}.csv')

	## visualizing result
	sns.heatmap(df)
	plt.savefig(f'{save_path}/wasserstein-{id}.png')
	plt.show()