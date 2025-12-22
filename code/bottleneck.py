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
	parser.add_argument('path', type=str)
	parser.add_argument('id', type=str)
	parser.add_argument('--save_path', type=str, default='../output')
	args = parser.parse_args()

	## defining variables
	path = args.path
	id = args.id
	save_path = args.save_path
	paths = os.listdir(path)
	bot = []

	## computing bottleneck distance
	for i in paths:
		bot_i = []
		dgm_i = np.load(f'{path}/{i}')
		for j in paths:
			if i != j:
				dgm_j = np.load(f'{path}/{j}')
				bot_i.append(bottleneck(dgm_i, dgm_j))
			else:
				bot_i.append(0)
		bot.append(bot_i)

	## saving wasserstein distance matrix
	df = pd.DataFrame(bot)
	df.to_csv(f'{save_path}/bottleneck-{id}.csv')

	## visualizing result
	sns.heatmap(df)
	plt.savefig(f'{save_path}/bottleneck-{id}.png')
	plt.show()