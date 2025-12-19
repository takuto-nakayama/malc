#  importing modules
from persim import bottleneck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, os


#  main process
if __name__ == '__main__':
	## parsing arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('path_h0', tyoe=str)
	parser.add_argument('path_h1', type=str)
	parser.add_argument('--save_path', type=str, default='../output/bottleneck.csv')
	args = parser.parse_args()

	## defining variables
	path_h0 = args.path_h0
	path_h1 = args.path_h1
	save_path = args.save_path
	paths_h0 = os.listdir(path_h0)
	paths_h1 = os.listdir(path_h1)
	bot = []

	## computing bottleneck distance
	for h0_i, h1_i in zip(paths_h0, path_h1):
		bot_i = []
		dgm_i = [np.load(h0_i), np.load(h1_i)]
		for h0_j, h1_j in zip(path_h0, path_h1):
			if h0_i != h0_j:
				dgm_j = [np.load(h0_j), np.load(h1_j)]
				bot_i.append(bottleneck(dgm_i, dgm_j))
			else:
				bot_i.append(0)
		bot.append(bot_i)

	## saving wasserstein distance matrix
	df = pd.DataFrame(bot)
	pd.to_csv(save_path)

	## visualizing result
	sns.heatmap(df)
	plt.savefig(f'{save_path[:-4]}.png')
	plt.show()