import os
import sys
from functions import *
from new_Models_and_Function import *
from numpy import *
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    

    ## SIMULATION OF DATA
    
    random.seed(1109992)
    
    
    N = 1000    
    C = 1
    processes = 10
    replications = 100
    
    ## CALCULATION
    
    graph_x = list()
    graph_y_prob = list()
    graph_y_dist = list()
    graph_x_support = list()
    
    for i in range(20):
        
        error = (i+1) / 10
        
        trainings_data = dataSimulation([0.8, 0.7, 0.9, -0.3], error, 0, N)
        prediction_data = dataSimulation([0.8, 0.7, 0.9, -0.3], error, 0, N)
        
        result = do_Boot(trainings_data, prediction_data, "linear", C, "auto", 1, processes, replications)
        result.view()
        print(i)
        graph_x.append(result.accuracy)
        graph_y_prob.append(result.var_probability)
        graph_y_dist.append(result.var_distance)
        
        graph_x_support.append(result.n_support[0]+result.n_support[1])
        
        
    ## PLOTTING
    def my_invert(matrix):
        return(list(zip(*matrix)))
    
    ## to sort
    multiple = [graph_x,graph_y_prob,graph_y_dist]
    multiple = my_invert(multiple)
    multiple = sorted(multiple)
    multiple = my_invert(multiple)
    graph_x,graph_y_prob,graph_y_dist = multiple
    ## end sort
        
    plt.plot(graph_x, graph_y_prob, c = 'black')
    plt.scatter(graph_x, graph_y_prob, c = 'black')
    plt.plot(graph_x, graph_y_dist, c = 'green')
    plt.scatter(graph_x, graph_y_dist, c = 'green')
    plt.show()
    
    
    ### Seond time with support vectors
    
    ## to sort
    multiple = [graph_x_support,graph_y_prob,graph_y_dist]
    multiple = my_invert(multiple)
    multiple = sorted(multiple)
    multiple = my_invert(multiple)
    graph_x,graph_y_prob,graph_y_dist = multiple
    ## end sort
        
    plt.plot(graph_x, graph_y_prob, c = 'black')
    plt.scatter(graph_x, graph_y_prob, c = 'black')
    plt.plot(graph_x, graph_y_dist, c = 'green')
    plt.scatter(graph_x, graph_y_dist, c = 'green')
    plt.show()
    
    