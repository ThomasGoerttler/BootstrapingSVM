import math
import statistics
    
def probability_distribution_of_points(array):
    "Swap the colums and the rows of the input matrix"
    return(list(zip(*array)))

def calculate_variance_of_svm(results):
    "results is an 2-dimensional array. It has N elements of array (of size m) which has a propability value of each point of the predciton dataset"
    
    # Swap the colums and the rows of the input matrix, so we have for each point the predicted values of each svm.
    distribution_of_points = probability_distribution_of_points(results)
    
    # Creating an empty array with size of points
    standard_deviations = [None] * len(distribution_of_points)
    
    # For each point the standard deviation of the different predicted values will be calculated
    for index in range(len(distribution_of_points)):
        std = statistics.stdev(distribution_of_points[index])
        standard_deviations[index] = std
        print(std)
        
    return(statistics.mean(standard_deviations))
