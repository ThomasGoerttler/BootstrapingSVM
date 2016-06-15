import math
import statistics
    

def calculate_variance_of_svm(results):
    
    # Creating an empty array with size of points
    standard_deviations = [None] * len(results)
    
    # For each point the standard deviation of the different predicted values will be calculated
    for index in range(len(results)):
        std = statistics.stdev(results[index])
        standard_deviations[index] = std
        print(index, ": ", std)
        
    return(statistics.mean(standard_deviations))
