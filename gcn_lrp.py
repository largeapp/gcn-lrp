# -*- coding:UTF-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np 
import cPickle as pkl
from tqdm import tqdm

class gcn_lrp(object):

	def __init__(self, support, gcn_output, gcn_input, weights, bias):
		self.support = support
		self.gcn_output = gcn_output
		self.gcn_input = gcn_input
		self.weights = weights
		self.bias = bias
		self.nodes_next = set([])
	
	def __call__(self, nodes):
		support = self.support
		gcn_output = self.gcn_output
		gcn_input = self.gcn_input
		weights = self.weights
		bias = self.bias
		nodes_next = self.nodes_next 
		
		if not bias:
			bias = [0 for _ in range(weights.shape[1])]
		support_gcn_input = np.matmul(support, gcn_input)
		R_final = np.zeros(gcn_input.shape)
		for node in nodes:
			z = np.matmul(support_gcn_input[node,:], weights) + bias
			s = gcn_output[node,:]/z
			c = np.matmul(s,weights.T)
			r = support_gcn_input[node,:]*c

			R = np.zeros(gcn_input.shape)

			support_vec = support[node,:]
			index = np.array(range(len(support_vec)))
			index_nonzero = index[support_vec != 0]
			nodes_next = nodes_next.union(set(list(index_nonzero)))

			input_vec = [gcn_input[node,:]*support_vec[j] for j in index_nonzero]
			input_vec_sum  = np.sum(input_vec,0)
			input_vec_sum += np.array([1e-9]*len(input_vec_sum))
			input_vec_division = [vec/input_vec_sum for vec in input_vec]

			for k in range(len(index_nonzero)):
				R[index_nonzero[k]] = r*input_vec_division[k]
			R_final += R
		self.nodes_next = nodes_next
		return R_final

class chebyshev_lrp(object):
	def __init__(self, support, gcn_output, gcn_input, weights, bias):
		self.support = support
		self.gcn_output = gcn_output
		self.gcn_input = gcn_input
		self.weights = weights
		self.bias = bias
		self.nodes_next = set([])
	
	def __call__(self, nodes):
		support = self.support
		gcn_output = self.gcn_output
		gcn_input = self.gcn_input
		weights = self.weights
		bias = self.bias
		nodes_next = self.nodes_next 
		mat_box = [np.matmul(support[i],np.matmul(gcn_input, weights[i])) for i in range(len(weights))]
		mat_sum = np.sum(mat_box,0)
		mat_box = [mat_box[i]/mat_sum for i in range(len(weights))]
		rs = [gcn_output*mat_box[i] for i in range(len(weights))]

		R_final = np.zeros(gcn_input.shape)
		for i in range(len(rs)):
			if len(bias) == 0:
				bias = np.zeros(weights[i].shape[1])
			g_lrp = gcn_lrp(support[i], rs[i], gcn_input, weights[i], bias)
			r = g_lrp(nodes)
			R_final += r
			nodes_next = nodes_next.union(g_lrp.nodes_next)
		self.nodes_next = nodes_next
		return R_final













