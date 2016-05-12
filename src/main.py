import os
import sys
from liblinearutil import *
from multiprocessing import Process, Pool
from bootstrapping import *
from confidence_calculation import *
from loadingData import *

if __name__ == '__main__':
    
    ### Prepare Data
    filename = sys.argv[1]
    print(filename)
    data = loadData(filename)
    
    ### Do bootstrapping
    PROCESSES = 10
    REPLICATIONS = 1000
    pool = Pool(processes = PROCESSES)
    results = pool.map(bootstrapping, [data] * REPLICATIONS, )
    
    ### Calculate Confidence Interavalls
    calculateConfidenceIntervalls(results)

