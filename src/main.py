import os
import sys
from multiprocessing import Process, Pool
from sample_svm import *
from confidence_calculation import *
from loading_data import *

if __name__ == '__main__':
    
    ### Prepare Data
    #filename = sys.argv[1]
    #print(filename)
    #data = load_data(filename)
    
    ### Simulation of the Data
    X = [[0, 0], [0, 1], [0, 0.5], [0, 1.5], [0, 12], [0, 13], [0, 0.25], [0, 12.5], [1, 1], [1, 0], [1, 1.5], [1, 0.5], [1, 11], [1, 2], [1, 13.5], [1, 0.52]]
    y = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    tranings_data = [y, X]
    
    X = [[0, 0], [0, 1], [0, 0.5], [0, 1.5], [0, 12], [0, 13], [0, 0.25], [0, 12.5], [1, 1], [1, 0], [1, 1.5], [1, 0.5], [1, 11], [1, 2], [1, 13.5], [1, 0.52]]
    y = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    prediction_data = [y, X]
    
    data = [tranings_data, prediction_data]
    
    ### Do bootstrapping
    PROCESSES = 10
    REPLICATIONS = 100
    pool = Pool(processes = PROCESSES)
    results = pool.map(single_sample_and_svm, [data] * REPLICATIONS)
    
    ### Calculate the Variance of the Support Vector Machine
    
    points_information = Points_Information(results)
    
    variance_of_svm_probabilites = calculate_variance_of_svm(points_information.probabilites)
    variance_of_svm_distance_to_hyperplane = calculate_variance_of_svm(points_information.distances)
    
    print("Variance of SVM Prob That 1")
    print(variance_of_svm_probabilites)
    
    print("Variance of SVM distance to hyperplane")
    print(variance_of_svm_distance_to_hyperplane)
    
        

