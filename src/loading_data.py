import os, sys
sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path 
from ctypes import c_double

def load_data(filename):
	
	data_file_name = './' + filename
	
	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)