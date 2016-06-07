import os
import sys
from liblinearutil import *
from multiprocessing import Process, Pool
from sample_svm import *
from confidence_calculation import *
from loading_data import *

if __name__ == '__main__':
    
    ### Prepare Data
    filename = sys.argv[1]
    print(filename)
    data = load_data(filename)
    
    ### Do bootstrapping
    PROCESSES = 10
    REPLICATIONS = 10#00
    pool = Pool(processes = PROCESSES)
    results = pool.map(single_sample_and_svn, [data] * REPLICATIONS)
    
    ### Calculate Confidence Interavalls
        
    calculate_confidence_intervalls(results)
    
    
    ### FAKE DATA
    ### N = 4
    ### size of prediction data = 8
    
    m1 = [0.20729282, 0.65201747, 0.55257748, 0.60321986, 0.47292820, 0.61189532, 0.32715436, 0.94512105]
    m2 = [0.22991870, 0.10980316, 0.93259272, 0.91276257, 0.94647900, 0.59004491, 0.22043538, 0.12538466]
    m3 = [0.22725206, 0.25272354, 0.25107548, 0.57239705, 0.73848822, 0.63439053, 0.50103990, 0.44679275]
    m4 = [0.23920063, 0.54783684, 0.20370402, 0.67215807, 0.62689568, 0.49102234, 0.71756493, 0.49745726]
    m5 = [1,1,1,1,1,1,1,1]
        
    results = [m1, m2, m3, m4]
    
    variance_of_svm = calculate_variance_of_svm(results)
    
    print("Variance of svm")
    print(variance_of_svm)
        

