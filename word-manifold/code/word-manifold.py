#  Word Manifold
#  computing Betti numbers from text data based on word n-grams. (cf. Fitz 2022)


#-----preprocess-----#
## importing modules
from datetime import datetime
from itertools import combinations, chain
from nltk import ngrams
from scipy.sparse import lil_matrix
import numpy as np
import pandas as pd
import argparse, csv, stanza


#-----main processes-----#
if __name__ == '__main__':
    start = datetime.now()

## parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('id', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('--lang_code', type=str, default='../data/stanza-langlist.csv')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--max_dim', type=int, default=4)
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
    lang_code = args.lang_code
    window_size = args.window_size
    max_dim = args.max_dim


    ## preparing tokenizer
    df_lang_code = pd.read_csv('/home/takuto/data/stanza-langlist.csv')[['Language', 'Icode']]
    iso = df_lang_code[df_lang_code['Language']==id]['Icode'].values[0]
    tokenizer = stanza.Pipeline(lang=iso, processors='tokenize')


    ## tokeinizing text
    with open(path, 'r') as file:
        doc = file.readlines()
        doc_sep = tokenizer.bulk_process(doc)

    token = []
    for line in doc_sep:
        token_in_line = []
        for sent in line.sentences:
            for word in sent.words:
                token_in_line.append(word.text)
        token.append(token_in_line)

    window = sorted(
        set(
            chain.from_iterable(
                [list(ngrams(tkns, n=window_size)) for tkns in tokens]
                )
            )
        )


    ## getting all skeleta
    skeleta = []

    for n in range(max_dim+1):
        Sn = []
        for w in window:
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


    ## computing Betti numbers
    betti = [id]
    for n in range(len(boundaries)):
        Bn = boundaries[n]
        dim_Cn = Bn.shape[1]
        nullity = dim_Cn - np.linalg.matrix_rank(Bn, 1e-10)
        if n+1 < len(boundaries):
            im_dim = np.linalg.matrix_rank(boundaries[n+1], 1e-10)
        else:
            im_dim = 0
    betti.append(nullity - im_dim)


    ## saving results
    with open(f'{save_path}', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(betti)
    for n,b in enumerate(betti[1:]):
        print(f'n={n}: Betti={b}')
    end = datetime.now()
    processing = (end - start).seconds
    print(f'{processing} seconds.')