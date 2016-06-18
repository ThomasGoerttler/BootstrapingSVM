import os
import sys
from multiprocessing import Process, Pool
from sample_svm import *
from confidence_calculation import *
from loading_data import *
from numpy import *
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    

    ## simple 
    
    N = 100
    
    x1 = random.sample(N) 
    x2 = random.sample(N) 
    
    xx = linspace(-0.1, 1.1)
    yy = xx
    
    
    y =  x1 - x2 
    adding = random.normal(0,0.5,N)
     
    y = y + adding
    y = sign(y)
    
    
    plt.scatter(x1,x2, c=y)
    
    ## REALLY INPORTANT
    
    X = list(zip(*[x1,x2]))
    
    #print(X)
    trainings_data = [y, X]
    prediction_data = [y, X]
    
    data = [trainings_data, prediction_data]
    
    kernel = 'linear'
    C = 100
    
    data = SVM_Input(trainings_data, prediction_data, kernel = kernel, C = C)
    
    real_result = do_svm(data)
    
    ### Do bootstrapping
    PROCESSES = 1
    REPLICATIONS = 100
    pool = Pool(processes = PROCESSES)
    results = pool.map(single_sample_and_svm, [data] * REPLICATIONS)
    
    ### Plot the hyperpyanes
    if(kernel == 'linear'):
        
        for i in range(len(yy)) :
            plt.plot(xx, results[i].line, '-k')
        plt.show()
    
    
    ### Calculate the Variance of the Support Vector Machine
    
    points_information = Points_Information(results)
    
    variance_of_svm_probabilites = calculate_variance_of_svm(points_information.probabilites)
    variance_of_svm_distance_to_hyperplane = calculate_variance_of_svm(points_information.distances)
    
    print("Variance of SVM Prob That 1")
    print(variance_of_svm_probabilites)
    
    print("Variance of SVM distance to hyperplane")
    print(variance_of_svm_distance_to_hyperplane)
    
    print("Number of Support Vectors")
    print(real_result.n_support)
    
    print("Accurancy")
    print(real_result.score)
    
    print("The C Factor")
    print(C)
    