#  Fitz (2025) Word Manifold
## importing modules
from itertools import combinations
from nltk import ngrams
from nltk.tokenize import word_tokenize
import numpy as np
import argparse, itertools


## parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('text', type=str)
parser.add_argument('--window_size', type=int, default=5)
args = parser.parse_args()


## defining variables
text = args.text
window_size = args.window_size
tokens = word_tokenize(text)
windows = list(
	ngrams(tokens, n=window_size)
	)


## getting all skeleta
skeleta = []

for n in range(window_size):
	Sn = []
	for w in windows:
		for idxs in combinations(range(len(w)), n+1):
			simplex = tuple(w[i] for i in idxs)
			Sn.append(simplex)

	skeleta.append(
		sorted(set(Sn))
		)


## getting boundary matrices
boundaries = []

for n in range(1, len(skeleta)):
    prev = skeleta[n-1]
    curr = skeleta[n]
    index_prev = {s: i for i, s in enumerate(prev)}
    Bn = np.zeros(
        (len(prev), len(curr)), dtype=int
            )
    
    for j, s in enumerate(curr):
        for i in range(len(s)):
            face = s[:i] + s[i+1:]
            sign = (-1)**i
            if face in index_prev:
                Bn[index_prev[face], j] += sign

    boundaries.append(Bn)


betti = []

for n in range(len(boundaries)):
    Bn = boundaries[n]

    # dim C_n = number of n-simplices
    dim_Cn = Bn.shape[1]

    # ker ∂n
    nullity = dim_Cn - np.linalg.matrix_rank(Bn, 1e-10)

    # im ∂n+1
    if n+1 < len(boundaries):
        im_dim = np.linalg.matrix_rank(boundaries[n+1], 1e-10)
    else:
        im_dim = 0

    betti_n = nullity - im_dim
    betti.append(betti_n)
    
for n,b in enumerate(betti):
	print(f'n={n}: Betti={b}')