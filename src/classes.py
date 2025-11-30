#  importing modules
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from transformers import BertTokenizer, BertModel

import numpy as np
import pandas as pd
import math, re, statistics, torch

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


	def get_sentence(self, token:str, num_text:int):
		list_filtered = []
		self.filtered = {}
		np.random.seed(42)
		id = self.tokenizer.convert_tokens_to_ids(token)
		idx = np.sort(np.random.choice(len(self.dataset['text']), num_text, replace=False))
		text = self.dataset['text'][idx]
		for t in text:
			t = re.sub(' *\n *', '\n', t)
			t = re.sub('\n\n+', '\n', t)
			t = t.split('\n')
			encoded = self.tokenizer(
				t,
				return_tensors='pt',
				padding=True,
				truncation=True,
				max_length=512
				)
			bool_mask = (encoded['input_ids']==id).any(dim=1)
			list_filtered.append({k: v[bool_mask] for k, v in encoded.items()})
		pad_id = self.tokenizer.pad_token_id
		self.filtered = self.pad_and_cat(list_filtered, pad_id)
		print(f'Sentences containing the word "{token}", ID {id}, have been extracted. ({len(self.filtered["input_ids"])} sentences)')



class Embedding:
	def __init__(self, model_name:str='bert-base-multilingual-cased', tokenizer_name:str='bert-base-multilingual-cased'):
		self.model = BertModel.from_pretrained(model_name)
		self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)


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
			
			processed = min(batch*(_+1), len(encoded['input_ids'])) * 100 // len(encoded['input_ids'])
			print(f'\rProcessing: {processed}%  |{"#"*(processed//4)}{"-"*(25-processed//4)}| ({min(batch*(_+1), len(encoded["input_ids"]))}/{len(encoded["input_ids"])})', end='', flush=True)
			
		self.output = torch.cat(list_batch, dim=0)



class Manifold:
	def metric(self, data:np.ndarray, point:np.ndarray, k:int, n:int):
		nn = NearestNeighbors(n_neighbors=k).fit(data)
		_, index = nn.kneighbors(point.reshape(1,-1))
		index = np.sort(index).squeeze()

		#  neighbors: k neighbor points from ``point``
		#  centered: the coordinate in which ``point`` is made to be the origin
		neighbors = data[index]  ## (k, 768)
		centered = neighbors - point  ## (k, 768)

		#  Vt: basis of the ``centered`` vector space
		#  rotated: vectors rotated with the basis
		U, S, Vt = np.linalg.svd(centered, full_matrices=False)  ## Vt: (k, 768)
		rotated = (centered @ Vt.T)[:,:n]  ## (k, n)

		poly = PolynomialFeatures(degree=3, include_bias=True)
		X_poly = poly.fit_transform(rotated)  ## (k, the number of arguments)
		feature_names = poly.get_feature_names_out()
		reg = LinearRegression(fit_intercept=False)
		reg.fit(X_poly, centered)
		coef = reg.coef_  ## (768, the number of arguments)

		index = [i for i, p in enumerate(poly.powers_) if sum(p) == 1]
		J = coef[:, index]  ## (768, n)
		g = J.T @ J  ##  (n, 768) @ (768, n) = (n, n)

		H = np.zeros((coef.shape[0], n, n))  ## (768, n, n)
		index = [i for i, p in enumerate(poly.powers_) if sum(p) == 2]
		for i in range(n):
			for j in range(n):
				for idx, power in zip(index, poly.powers_[index]):
					deno = 1
					for p in power:
						deno *= math.factorial(p)
					H[:,i,j] = coef[:,idx] * deno

		## (i,j): 
		H[:, i, j] = coef[:, idx] * 2 ## (coefs:768, an argument:n, another argument:n)

		H3 = np.zeros((coef.shape[0], n, n, n))
		index = [i for i, p in enumerate(poly.powers_) if sum(p) == 3]
		for i in range(n):
			for j in range(n):
				for k in range(n):
					for idx, power in zip(index, poly.powers_[index]):
						deno = 1
						for p in power:
							deno *= math.factorial(p)
						H3[:, i, j, k] = coef[:, idx] * deno ## (coefs:768, n, n, n)


		#  dg[k,i,j] = <H[:,k,i], J[:,j]⟩+⟨J[:,i], H[:,k,j]>
		#  \frac{\partial}{\partial x_k} g_{ij}
		dg = np.zeros((n, n, n))  ## (n, n, n)
		dg = np.einsum('pik,pj->kij', H, J) + np.einsum('pj,pik->kij', J, H)
		
		#  d2g[l,k,i,j] = \\partial_l \partial_m g_{ij}
		#  = <H3[:,l,k,i], J[:,j]> + <H[:,k,i], H[:,l,j]> + <H[:,l,i], H[:,k,j]> + <J[:,i], H3[:,l,k,j]>
		d2g = (
			np.einsum('plki,pj->lkij', H3, J) + 
			np.einsum('pki,plj->lkij', H, H) + 
			np.einsum('pli,pkj->lkij', H, H) + 
			np.einsum('pi,plkj->lkij', J, H3)
		)

		return g, dg, d2g


	def christoffel(self, g, dg, d2g):
		g_inv = np.linalg.inv(g)  ## (n,n)

		#  dg = \frac{\partial}{\partial x_k} g_{ij} -> [k,i,j]
		#  [j,k,l]: transformed from [k,i,j] into for \Gamma
		#  term = [j,k,l]+[k,j,l]-[l,j,k] -> (0,1,2)+(1,0,2)-(2,0,1)
		#  gamma = 0.5 * g_inv * term -> [i,l],[j,k,l]
		term = (
			dg +
			np.transpose(dg, (1,0,2)) -
			np.transpose(dg, (2,0,1))
		)
		gamma = 0.5 * np.einsum('il, jkl -> ijk', g_inv, term)  ## (n,n,n)

		n = g.shape[0]
		dg_inv = np.zeros((n, n, n))
		#  ∵ g^{-1}g = I -> \frac{\partial}{\partial_m}g^{-1}g = 0
		for m in range(n):
			dg_inv[m] = - g_inv @ dg[m] @ g_inv
		
		dgamma = np.zeros((n, n, n, n))
		tmp = (  ## [m,k,i,j] = \partial_m \partial_k g_{ij}
			d2g +
			np.transpose(d2g, (0,2,1,3)) -
			np.transpose(d2g, (0,3,1,2))
		)
		term1 = 0.5 * np.einsum('mil,jkl->mijk', dg_inv, term)
		term2 = 0.5 * np.einsum('il,mjkl->mijk', g_inv, tmp)
		
		dgamma = term1 + term2  ## [m,i,j,k] = \partial_m \gamma^i_{jk}

		return gamma, dgamma


	def curvature_tensor(self, data:np.ndarray, point:np.ndarray, k:int, n:int):
		g, dg, d2g = self.metric(data, point, k, n)
		gamma, dgamma = self.christoffel(g, dg, d2g)
		R = np.zeros((n, n, n, n))
		
		## term [l,i,j,k] = \partial_k \gamma^i_{lj} - \partial_l \gamma^i_{kj}
		term = np.transpose(dgamma, (1,0,3,2)) - np.transpose(dgamma, (0,1,2,3))
		s1 = np.einsum('ikm,mlj->ikjl', gamma, gamma)  ## s1 = \gamma^i_{km} \gamma^m_{lj}
		s2 = np.einsum('ilm,mkj->ikjl', gamma, gamma)  ## s2 = \gamma^i_{lm} \gamma^m_{kj}
		R = term + s1 - s2

		return R
