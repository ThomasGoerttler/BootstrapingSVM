import os
import sys
from functions import *
from numpy import *
import math



if __name__ == '__main__':
    

    ## SIMULATION 
    
    N = 1000
    
    x1 = random.sample(N) 
    x2 = random.sample(N) 
    
    y =  x1 - x2 
    adding = random.normal(0,0.5,N)
     
    #y = y + adding
    y = sign(y)
    
    
    ## REALLY INPORTANT
    X = list(zip(*[x1,x2]))
    
    #print(X)
    trainings_data = [y, X]
    prediction_data = [y, X]
    kernel = 'poly'
    C = 100
    processes = 10
    replications = 100
    
    result = do_Bootstrap(trainings_data, prediction_data, "linear", C, "auto", 1, processes, replications)
    print(result)
    print(result[0].n_support_)
    
    result = do_Bootstrap(trainings_data, prediction_data, "poly", C, "auto", 5, processes, replications)
    print(result)
    
    result = do_Bootstrap(trainings_data, prediction_data, "poly", C, "auto", 10, processes, replications)
    print(result)
    
    result = do_Bootstrap(trainings_data, prediction_data, "rbf", C, "auto", 5, processes, replications)
    print(result)
    