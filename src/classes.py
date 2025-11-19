#  importing modules
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from transformers import BertTokenizer, BertModel

import numpy as np
import pandas as pd
import os
import re
import torch


class Wiki:
	def __init__(self, lang:str):
		self.dataset = load_dataset('wikimedia/wikipedia', f'20231101.{lang}')['train']
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


	def pad_and_cat(self, list_filtered, pad_id):
		output = {}
		keys = list_filtered[0].keys()
		for k in keys:
			parts = [d[k] for d in list_filtered if d[k].size(0) > 0]
			if not parts:
				pseudo = list_filtered[0][k]
				output[k] = pseudo.new_empty((0, 0))
				continue
			max_len = max(t.size(1) for t in parts)
			if k == 'input_ids':
				pad_val = pad_id
			elif k in ('attention_mask', 'token_type_ids'):
				pad_val = 0
			else:
				pad_val = 0
			parts = [torch.nn.functional.pad(t, (0, max_len - t.size(1)), value=pad_val) for t in parts]
			output[k] = torch.cat(parts, dim=0)
		return output


	def get_sentence(self, token:int, text_range:tuple):
		list_filtered = []
		self.filtered = {}
		id = self.tokenizer.convert_tokens_to_ids(token)
		for i in range(text_range[0], text_range[1]):
			text = re.sub(' *\n *', '\n', self.dataset[i]['text'])
			text = re.sub('\n\n+', '\n', text)
			text = text.split('\n')
			encoded = self.tokenizer(
				text,
				return_tensors='pt',
				padding=True,
				truncation=True,
				max_length=512
				)
			bool_mask = (encoded['input_ids']==id).any(dim=1)
			list_filtered.append({k: v[bool_mask] for k, v in encoded.items()})
		pad_id = self.tokenizer.pad_token_id
		self.filtered = self.pad_and_cat(list_filtered, pad_id)
		print(f'Sentences containing the word ID {id} have been extracted. ({len(self.filtered["input_ids"])} sentences)')



class Embedding:
	def __init__(self):
		self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


	def embed(self, token:str, encoded:dict, batch:int):
		if torch.cuda.is_available():
			device = torch.device('cuda')	
			print(f'devide in use: cuda')		
		elif torch.backends.mps.is_available():
			device = torch.device('mps')
			print(f'devide in use: mps')		
		else:
			device = 'cpu'
			print(f'devide in use: cpu')

		id = self.tokenizer.convert_tokens_to_ids(token)
		self.model.to(device).eval()
		encoded = {k: v.to(device) for k, v in encoded.items()}
		list_batch = []
		cnt_batch = len(encoded['input_ids']) // batch

		for _ in range(cnt_batch):
			encoded_batch = {k:encoded[k][batch*_:min(batch*(_+1), len(encoded[k]))] for k in encoded.keys()}
			mask = (encoded_batch['input_ids'] == id)
			s_idx, t_idx = mask.nonzero(as_tuple=True)
			
			with torch.inference_mode():
				output_batch = self.model(**encoded_batch)['last_hidden_state']
				list_batch.append(output_batch[s_idx, t_idx].to('cpu'))
			
			processed = min(batch*(_+1), len(encoded['input_ids'])) * 100 // len(encoded["input_ids"])
			print(f'\rProcessing: {processed}%  |{"#"*(processed//4)}{"-"*(25-processed//4)}| ({min(batch*(_+1), len(encoded["input_ids"]))}/{len(encoded["input_ids"])})', end='', flush=True)
			
		self.output = torch.cat(list_batch, dim=0)



class Manifold:
	def metric(self, data:np.ndarray, point:np.ndarray, k:int, n:int):
		nn = NearestNeighbors(n_neighbors=k).fit(data)
		_, index = nn.kneighbors(point.reshape(1,-1))
		index = np.sort(index).squeeze()
		neighbors = data[index]
		centered = neighbors - point
		U, S, Vt = np.linalg.svd(centered, full_matrices=False)
		rotated = (centered @ Vt.T)[:,:n]
		poly = PolynomialFeatures(degree=2, include_bias=True)
		X_poly = poly.fit_transform(rotated)
		feature_names = poly.get_feature_names_out()
		reg = LinearRegression(fit_intercept=False)
		reg.fit(X_poly, centered)
		coef = reg.coef_      

		g_idx = []
		for i in range(n):
			name = f"x{i}"
			g_idx.append(int(np.where(feature_names == name)[0][0]))
		J = coef[:, g_idx]
		g = J.T @ J

		H = np.zeros((coef.shape[0], n, n))
		for i in range(n):
			for j in range(n):
				if i == j:
					name = f"x{i}^2"
				else:
					name = f"x{min(i,j)} x{max(i,j)}"

				idx = np.where(feature_names == name)[0][0]
				H[:, i, j] = coef[:, idx]

		dg = np.zeros((n, n, n))
		dg = np.einsum('pik,pj->kij', H, J) + np.einsum('pj,pik->kij', J, H)
		
		return g, dg, J, H


	def christoffel(self, g, dg, H):
		g_inv = np.linalg.inv(g)
		term = dg + np.transpose(dg, (1,0,2)) - np.transpose(dg, (2,0,1))
		gamma = 0.5 * np.einsum('il, jkl -> ijk', g_inv, term)

		n = g.shape[0]
		dg_inv = np.zeros((n, n, n))
		for m in range(n):
			dg_inv[m] = - g_inv @ dg[m] @ g_inv
		d2g = np.zeros((n, n, n, n))
		d2g = (
			np.einsum('dim,dkj->mjik', H, H) +
			np.einsum('dij,dkm->mjik', H, H)
		)
		T = np.zeros((n, n, n))
		T = dg + np.transpose(dg, (2,1,0)) - np.transpose(dg, (1,0,2))
		dgamma = np.zeros((n, n, n, n))
		tmp = d2g + np.transpose(d2g, (0,2,1,3)) - np.transpose(d2g, (0,2,1,3))
		term1 = 0.5 * np.einsum('mil,ljk->mijl', dg_inv, T)
		term2 = 0.5 * np.einsum('il,mjlk->mijk', g_inv, tmp)
		dgamma = term1 + term2

		return gamma, dgamma


	def curvature_tensor(self, gamma, dgamma):
		n = gamma.shape[0]
		R = np.zeros((n, n, n, n))
		term = np.transpose(dgamma, (1,2,0,3)) - np.transpose(dgamma, (1,0,2,3))
		s1 = np.einsum('mik,ljm->lijk', gamma, gamma)
		s2 = np.einsum('mjk,lim->lijk', gamma, gamma)
		R = term + s1 - s2

		return R
