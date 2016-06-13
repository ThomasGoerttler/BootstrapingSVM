import os
import sys
from multiprocessing import Process, Pool
from sample_svm import *
from confidence_calculation import *
from loading_data import *

if __name__ == '__main__':
    
    ### Prepare Data
    filename = sys.argv[1]
    print(filename)
    data = load_data(filename)
    
    ### FAKE DATA
    X = [[0, 0], [0, 1], [0, 0.5], [0, 1.5], [0, 12], [0, 13], [0, 0.25], [0, 12.5], [1, 1], [1, 0], [1, 1.5], [1, 0.5], [1, 11], [1, 2], [1, 13.5], [1, 0.52]]
    y = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    data = [y, X]
    
    ### Do bootstrapping
    PROCESSES = 10
    REPLICATIONS = 1000
    pool = Pool(processes = PROCESSES)
    results = pool.map(single_sample_and_svm, [data] * REPLICATIONS)
    
    ### Calculate Confidence Interavalls
    
    variance_of_svm = calculate_variance_of_svm(results)
    
    print("Variance of SVM")
    print(variance_of_svm)
        

