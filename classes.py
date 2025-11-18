#  importing modules
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel

import numpy as np
import pandas as pd
import os
import re
import torch
import zulip


class Wiki:
	def __init__(self, lang:str):
		self.dataset = load_dataset('wikimedia/wikipedia', f'20231101.{lang}')['train']
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


	def _pad_and_cat(self, list_filtered, pad_id):
		out = {}
		keys = list_filtered[0].keys()
		for k in keys:
			parts = [d[k] for d in list_().filtered if d[k].size(0) > 0]
			if not parts:
				proto = list_filtered[0][k]
				out[k] = proto.new_empty((0, 0))
				continue

			max_len = max(t.size(1) for t in parts)
			if k == 'input_ids':
				pad_val = pad_id
			elif k in ('attention_mask', 'token_type_ids'):
				pad_val = 0
			else:
				pad_val = 0

			parts = [torch.nn.functional.pad(t, (0, max_len - t.size(1)), value=pad_val) for t in parts]
			out[k] = torch.cat(parts, dim=0)
		return out


	def get_sentence(self, id:int, text_range:tuple):
		list_filtered = []
		self.filtered = {}
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
		self.filtered = self._pad_and_cat(list_filtered, pad_id)
		print(f'Sentences containing the word ID {id} have been extracted. ({len(self.filtered["input_ids"])} sentences)')



class Embedding:
	def __init__(self):
		self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


	def embed(self, id:int, encoded:dict, batch_size:int):
		if torch.cuda.is_available():
			device = torch.device('cuda')	
			print(f'devide in use: cuda')		
		elif torch.backends.mps.is_available():
			device = torch.device('mps')
			print(f'devide in use: mps')		
		else:
			device = 'cpu'
			print(f'devide in use: cpu')

		self.model.to(device).eval()
		encoded = {k: v.to(device) for k, v in encoded.items()}
		list_batch = []
		batch_num = len(encoded['input_ids']) // batch_size

		for _ in range(batch_num):
			encoded_batch = {k:encoded[k][batch_size*_:min(batch_size*(_+1), len(encoded[k]))] for k in encoded.keys()}
			mask = (encoded_batch['input_ids'] == id)
			s_idx, t_idx = mask.nonzero(as_tuple=True)
			
			with torch.inference_mode():
				output_batch = self.model(**encoded_batch)['last_hidden_state']
				list_batch.append(output_batch[s_idx, t_idx].to('cpu'))
			
			processed = min(batch_size*(_+1), len(encoded['input_ids'])) * 100 // len(encoded["input_ids"])
			print(f'\rProcessing: {processed}%  |{"#"*(processed//4)}{"-"*(25-processed//4)}| ({min(batch_size*(_+1), len(encoded["input_ids"]))}/{len(encoded["input_ids"])})', end='', flush=True)
			
		self.output = torch.cat(list_batch, dim=0)



class Manifold:
	def metric(self, data:np.ndarray, point:np.ndarray, k:int, n:int):
		nn = NearestNeighbors(n_neighbors=k)
		nn.fit(data)

		_, index = nn.kneighbors(point.reshape(1,-1))
		neighbors = data[index].squeeze()
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
				# 対称性のため min/max を使う
				if i == j:
					name = f"x{i}^2"
				else:
					name = f"x{min(i,j)} x{max(i,j)}"

				idx = np.where(feature_names == name)[0][0]
				H[:, i, j] = coef[:, idx]

		dg = np.zeros((n, n, n))
		for k in range(n):
			for i in range(n):
				for j in range(n):
					dg[k, i, j] = np.sum(
						H[:, i, k] * J[:, j] +
						J[:, i] * H[:, j, k])
		
		return g, dg, J, H


	def christoffel(self, g, dg, J, H):
		g_inv = np.linalg.inv(g)
		term = dg + np.transpose(dg, (1,0,2)) - np.transpose(dg, (2,0,1))
		gamma = 0.5 * np.einsum('il, jkl -> ijk', g_inv, term)
		
		n = g.shape[0]
		# ---- (1) inverse metric derivative: dg_inv[m,i,l] ----
		dg_inv = np.zeros((n, n, n))
		for m in range(n):
			dg_inv[m] = - g_inv @ dg[m] @ g_inv

		# ---- (2) second derivatives of g: d2g[m,j,i,k] = ∂_m ∂_j g_{ik} ----
		d2g = np.zeros((n, n, n, n))
		# Using formula with H
		# ∂_m∂_j g_{ik} = sum_d (H[d,i,m] H[d,k,j] + H[d,i,j] H[d,k,m])
		d2g = (
			np.einsum('dim,dkj->mjik', H, H) +
			np.einsum('dij,dkm->mjik', H, H)
		)

		# ---- (3) build T[j,l,k] ----
		T = np.zeros((n, n, n))
		for j in range(n):
			for l in range(n):
				for k in range(n):
					T[j,l,k] = dg[j,l,k] + dg[k,l,j] - dg[l,j,k]

		# ---- (4) derivative of christoffel: dgamma[m,i,j,k] ----
		dgamma = np.zeros((n, n, n, n))
		for m in range(n):
			for i in range(n):
				for j in range(n):
					for k in range(n):
						# first term: 0.5 * (∂_m g^{i l}) T[j,l,k]
						term1 = 0.5 * np.sum(dg_inv[m,i,:] * T[j,:,k])

						# second term:
						# 0.5 * g^{i l} ( ∂_m∂_j g_{l k} + ∂_m∂_k g_{l j} - ∂_m∂_l g_{j k} )
						tmp = (
							d2g[m, j, :, k] +
							d2g[m, k, :, j] -
							d2g[m, :, j, k]
						)
						term2 = 0.5 * np.sum(g_inv[i,:] * tmp)

						dgamma[m,i,j,k] = term1 + term2

		return gamma, dgamma


	def curvature_tensor(self, gamma, dgamma):
		n = gamma.shape[0]
		R = np.zeros((n, n, n, n))
		for l in range(n):
			for i in range(n):
				for j in range(n):
					for k in range(n):
						term = dgamma[j, l, i, k] - dgamma[i, l, j, k]
						s1 = 0.0
						s2 = 0.0
						for m in range(n):
							s1 += gamma[m, i, k] * gamma[l, j, m]
							s2 += gamma[m, j, k] * gamma[l, i, m]
						R[l, i, j, k] = term + s1 - s2
		
		return R


class Zulip:
	def __init__(self):
		load_dotenv()
		email = os.getenv('EMAIL')
		api_key = os.getenv('API_KEY')
		site = os.getenv('SITE')
		self.to = os.getenv('TO')
		self.topic = os.getenv('TOPIC')
		self.client = zulip.Client(
			email=email,
			api_key=api_key,
			site=site
		)


	def send_report(self, content):
		self.client.send_message({
			"type": 'steam',
			"to": self.to,
			"topic": self.topic,
			"content": content
		})
