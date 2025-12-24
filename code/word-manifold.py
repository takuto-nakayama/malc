#  Fitz (2025) Word Manifold
## importing modules
from itertools import combinations, chain
from nltk import ngrams
from nltk.tokenize import word_tokenize
from scipy.sparse import lil_matrix
import numpy as np
import argparse, csv


## parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('id', type=str)
parser.add_argument('save_path', type=str)
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--max_dim', type=int, default=2)
args = parser.parse_args()

def rank_mod2_sparse(B):
    """
    Compute rank of a sparse matrix over Z_2.
    """
    B = B.copy().tolil()
    rows, cols = B.shape
    r = 0

    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if B[i, c] % 2 != 0:
                pivot = i
                break
        if pivot is None:
            continue

        # swap rows
        if pivot != r:
            B.rows[r], B.rows[pivot] = B.rows[pivot], B.rows[r]
            B.data[r], B.data[pivot] = B.data[pivot], B.data[r]

        # eliminate (XOR)
        for i in range(rows):
            if i != r and B[i, c] % 2 != 0:
                B[i, :] = B[i, :] + B[r, :]
                B.data[i] = [x % 2 for x in B.data[i]]

        r += 1
        if r == rows:
            break

    return r



## defining variables
path = args.path
id = args.id
save_path = args.save_path
window_size = args.window_size
max_dim = args.max_dim

with open(path, 'r') as f:
    text = f.readlines()
tokens = [word_tokenize(sent) for sent in text]
windows = [list(ngrams(tkns, n=window_size)) for tkns in tokens]
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
boundaries = [lil_matrix(
        (1, len(skeleta[0])),
        dtype=int
        )]
boundaries[0][0,:] = [
    1 for _ in range(len(skeleta[0]))
    ]
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
            Bn[index_prev[hat], j] += 1
    boundaries.append(Bn.tocsr())


betti = [id]
for n in range(len(boundaries)):
    Bn = boundaries[n]
    rank_n = rank_mod2_sparse(Bn)
    if n + 1 < len(boundaries):
        rank_np1 = rank_mod2_sparse(boundaries[n + 1])
    else:
        rank_np1 = 0
    betti_n = Bn.shape[1] - rank_n - rank_np1
    betti.append(betti_n)


with open(save_path, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(betti)
for n,b in enumerate(betti):
	print(f'n={n}: Betti={b}')