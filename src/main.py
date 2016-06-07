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
    REPLICATIONS = 1000
    pool = Pool(processes = PROCESSES)
    results = pool.map(single_sample_and_svn, [data] * REPLICATIONS)
    
    ### Calculate Confidence Interavalls
    calculate_confidence_intervalls(results)

