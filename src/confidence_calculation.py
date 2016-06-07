import math
import statistics


def calculate_confidence_intervalls(results):

    accuracies = []
    for i in range(len(results)):
        accuracies.append(results[i][0])
    meanAccuracy = statistics.mean(accuracies)
    standardDeviation = statistics.stdev(accuracies)
    
    print("Mean accuracy: " + str(meanAccuracy))
    print("Confidence: [" + str(meanAccuracy - 1.96 * standardDeviation) + ", " + str(meanAccuracy + 1.96 * standardDeviation) +"]")
    
    
    
def probability_distribution_of_points(array):
    return(list(zip(*array)))

def calculate_variance_of_svm(results):
    "results is an 2-dimensional array. It has N elements of array (of size m) which has a propability value of each point of the predciton dataset"
    distribution_of_points = probability_distribution_of_points(results)
    
    standard_deviations = [None] * len(distribution_of_points)
    
    for index in range(len(distribution_of_points)):
        std = statistics.stdev(distribution_of_points[index])
        standard_deviations[index] = std
        
    return(statistics.mean(standard_deviations))
