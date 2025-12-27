from datetime import datetime
from itertools import combinations, chain
from nltk import ngrams
from scipy.sparse import lil_matrix
import numpy as np
import pandas as pd
import stanza


class WordManifold:
	def __init__(self, lang, text_path, window_size):
		self.lang = lang
		self.window_size = window_size

		# preparing tokenizer
		df_lang_code = pd.read_csv('resourse/stanza-langlist.csv')
		code = df_lang_code[df_lang_code['language']==lang]['code'].values[0]
		tokenizer = stanza.Pipeline(lang=code, processors='tokenize')
		# tokening each lines
		with open(text_path, 'r') as file:
			self.doc = file.readlines()
			self.doc_sep = tokenizer.bulk_process(self.doc)

		# inducing each word
		self.token = []
		for line in self.doc_sep:
			token_in_line = []
			for sent in line.sentences:
				for word in sent.words:
					token_in_line.append(word.text)
			self.token.append(token_in_line)


	def get_ngram(self):
		return sorted(
		set(
			chain.from_iterable(
				[list(ngrams(tkns, n=self.window_size)) for tkns in self.token]
				)
			)
		)


	def get_skeleta(self, ngram):
		skeleta = []
		for n in range(self.window_size+1):
			Sn = []
			for w in ngram:
				for idxs in combinations(range(len(w)), n+1):
					simplex = tuple(w[i] for i in idxs)
					Sn.append(simplex)
			skeleta.append(
				sorted(set(Sn))
				)
			
		return skeleta
	

	def get_boundary(self, skeleta):
		# making a list for the output
		B = [lil_matrix(
			(1, len(skeleta[0])),
			dtype=int
			)]
		B[0][0,:] = [
		1 for _ in range(len(skeleta[0]))
		]

		# Bn: boundary matrices for Sn in a sparse matrices
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
			B.append(Bn.tocsr())

		return B


	def get_betti(self, B):
		betti = [id]
		for n in range(len(B)):
			Bn = B[n]
			dim_Cn = Bn.shape[1]
			nullity = dim_Cn - np.linalg.matrix_rank(Bn, 1e-10)
			if n+1 < len(B):
				dim_im = np.linalg.matrix_rank(B[n+1], 1e-10)
			else:
				dim_im = 0
		betti.append(nullity - dim_im)

		return betti
