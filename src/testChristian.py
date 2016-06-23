import os
import sys
from functions import *
from numpy import *
import math
from numpy import random as rd
from matplotlib import pyplot as plot

N = 1000
Kernel = "rbf"
Gamma = "auto"
processes = 1
replications = 50
Degree = 1



trainingsdata = dataSimulation([.5,.25,.25],1, 1,N)

testdata = dataSimulation([.5,.25,.25],1, 1, N)


CValues = list(arange(0.1,1,0.1))
results = []
variances = []
accuracies = []
SVnumbers = []
for i in range(len(CValues)):
	C = CValues[i]
	result = do_Bootstrap(trainingsdata, testdata, Kernel, C, Gamma, Degree, processes, replications)
	results = results + result
	variances = variances + [result[2]]
	accuracies = accuracies + [result[1]]
	SVnumbers = SVnumbers + [list(result[0].n_support_)[0]]
	print(result)
evaluation = list(zip(*[CValues, SVnumbers, accuracies, variances]))
for i in range(len(evaluation)):
	print(evaluation[i])
	
Test = centroidSimulation((1,-1),[(-1,-1),(1,1)], 0, 2000, 0, 'euclidean')
plotData = list(zip(*Test[1]))
plot.scatter(plotData[0],plotData[1], c=Test[0])
plot.show()

Test = centroidSimulation((1,-1, .5),[(-2,-2),(1,2), (3,0)], .1, 10000, 0, 'euclidean', "uniform", par1 = -5, par2 = 5)
plotData = list(zip(*Test[1]))
plot.scatter(plotData[0],plotData[1], c=Test[0])
plot.show()

Test = centroidSimulation((1),[(0,0)], 0, 10000, -1, 'euclidean', "uniform", par1 = -5, par2 = 5)
plotData = list(zip(*Test[1]))
plot.scatter(plotData[0],plotData[1], c=Test[0])
plot.show()	