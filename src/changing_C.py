import os
import sys
from functions import *
from new_Models_and_Function import *
from numpy import *
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    

    ## SIMULATION OF DATA
    
    random.seed(11091992)
    
    
    N = 1000
    
    trainings_data = dataSimulation([0.8, 0.7, 0.9, -0.3], 1, 0, N)
    prediction_data = dataSimulation([0.8, 0.7, 0.9, -0.3], 1, 0, N)
    
    C = 0.1
    processes = 10
    replications = 100
    
    
    ## CALCULATION
    
    graph_x = list()
    graph_y_prob = list()
    graph_y_dist = list()
    
    for i in range(20):
        C = (i+1) / 10
        result = do_Boot(trainings_data, prediction_data, "linear", C, "auto", 1, processes, replications)
        result.view()
        print(i)
        graph_x.append(C)
        graph_y_prob.append(result.var_probability)
        graph_y_dist.append(result.var_distance)
        
        
    ## PLOTTING
        
    plt.plot(graph_x, graph_y_prob, c = 'black')
    plt.scatter(graph_x, graph_y_prob, c = 'black')
    plt.plot(graph_x, graph_y_dist, c = 'green')
    plt.scatter(graph_x, graph_y_dist, c = 'green')
    plt.show()
        
    