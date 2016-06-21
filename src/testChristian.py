import os
import sys
from functions import *
from numpy import *
import math

N = 1000
Kernel = "linear"
Gamma = "auto"
processes = 1
replications = 50
Degree = 1



trainingsdata = dataSimulation([.5,.25,.25],1, 1,N)

testdata = dataSimulation([.5,.25,.25],1, 1, N)


CValues = list(arange(0.1,2,0.1))
results = []
variances = []
SVnumbers = []
for i in range(len(CValues)):
	C = CValues[i]
	result = do_Bootstrap(trainingsdata, testdata, Kernel, C, Gamma, Degree, processes, replications)
	results = results + result
	variances = variances + [result[2]]
	SVnumbers = SVnumbers + [list(result[0].n_support_)[0]]
	print(result)
print(CValues, SVnumbers, variances)