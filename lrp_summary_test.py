from __future__ import division
from __future__ import print_function
from gcn_lrp import *
import numpy as np 
import cPickle as pk 
from tqdm import tqdm

model_param_path = './model_param/'

with open(model_param_path+'support', 'r') as f:
	support = pk.load(f)

with open(model_param_path+'weights', 'r') as f:
	wl = pk.load(f)

with open(model_param_path+'activaties', 'r') as f:
	a = pk.load(f)

name_str = 'test'
with open(model_param_path+'y_'+name_str, 'r') as f:
	y = pk.load(f)
y_dim = len(y[0])

with open(model_param_path+name_str+'_mask', 'r') as f:
	mask = pk.load(f)

with open(model_param_path+'bias', 'r') as f:
	bias = pk.load(f)

y_nunzeor = (np.array(range(len(mask))))[mask != 0]


L = len(y_nunzeor)

dic1, dic2, dic3 = {},{},{}
for i in tqdm(range(L)):
	node = y_nunzeor[i]
	l1, l2, l3 = {}, {}, {}
	y_ = (np.array(range(y_dim)))[y[node] != 0]
	r0 = a[-1]
	nodes = [node]
	for j in range(1,len(a)):
		lrp = gcn_lrp(support, r0, a[-j-1], wl[-j], bias)
		r0 = lrp(nodes)
		nodes = lrp.nodes_next

	r00 = r0*r0
	r0_sum = np.sqrt(np.sum(r00, 1))
	index = (np.array(range(r0_sum.shape[0])))[r0_sum != 0]
	sum_ = np.sum(r0_sum)
	for x in index:
		l1[x] = r0[x, :]
		l2[x] = r0_sum[x]/sum_				
	dic1[node] = l1
	dic2[node] = l2			


summary_save_path = './lrp_result/'

data_name = 'cora'
version = '.notonlyrightlabel'

with open(summary_save_path+data_name+".16."+name_str+".dic1"+version, 'w+') as f:
	pk.dump(dic1,f)
with open(summary_save_path+data_name+".16."+name_str+".dic2"+version, 'w+') as f:
	pk.dump(dic2,f)
