#  Fitz (2025) Word Manifold
## importing modules
from itertools import combinations, chain
from nltk import ngrams
from nltk.tokenize import word_tokenize
from scipy.sparse import lil_matrix
import numpy as np
import argparse


## parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--max_dim', type=int, default=2)
args = parser.parse_args()


## defining variables
path = args.path
window_size = args.window_size
max_dim = args.max_dim

with open(path, 'r') as f:
    text = f.readlines()
tokens = [word_tokenize(sent) for sent in text]
windows = [list(ngrams(tkns, n=5)) for tkns in tokens]
windows = sorted(set(chain.from_iterable(windows)))


## getting all skeleta
skeleta = []

for n in range(max_dim+1):
	Sn = []
	for w in windows:
		for idxs in combinations(range(len(w)), n+1):
			simplex = tuple(w[i] for i in idxs)
			Sn.append(simplex)

	skeleta.append(
		sorted(set(Sn))
		)


## getting boundary matrices
boundaries = [[1 for _ in range(len(skeleta[0]))]]
for n in range(1, len(skeleta)):
    prev = skeleta[n-1]
    curr = skeleta[n]
    index_prev = {s: i for i, s in enumerate(prev)}
    Bn = lil_matrix(
        (len(prev), len(curr)), dtype=int
            )
    
    for j, s in enumerate(curr):
        for i in range(len(s)):
            hat = s[:i] + s[i+1:]
            val = (-1)**i
            Bn[index_prev[hat], j] += val

    boundaries.append(Bn.tocsr())


betti = []
for n in range(len(boundaries)):
    Bn = boundaries[n]
    rank_n = np.linalg.matrix_rank(boundaries[n].toarray())
    if n+1 < len(boundaries):
        rank_np1 = np.linalg.matrix_rank(boundaries[n+1].toarray())
    else:
        rank_np1 = 0
    betti_n = boundaries[n].shape[1] - rank_n - rank_np1
    betti.append(betti_n)
    
for n,b in enumerate(betti):
	print(f'n={n}: Betti={b}')