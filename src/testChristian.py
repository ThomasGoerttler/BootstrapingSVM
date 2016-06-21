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


x1 = random.standard_normal(N)
x2 = random.standard_normal(N)
x3 = random.standard_normal(N)
y = 0.5*x1+0.25*x2+0.25*x3 + random.standard_normal(N)
y = sign(y)

X = list(zip(*[x1,x2,x3]))
trainingsdata = [y,X]

x1 = random.standard_normal(N)
x2 = random.standard_normal(N)
x3 = random.standard_normal(N)
y = 0.5*x1+0.25*x2+0.25*x3 + random.standard_normal(N)
y = sign(y)

X = list(zip(*[x1,x2,x3]))
testdata = [y,X]


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