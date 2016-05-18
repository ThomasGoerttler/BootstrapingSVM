import math

def mean(array):
    sum = 0
    for i in range(len(array)):
        sum += array[i]
    return (sum/len(array))
    
def standard_deviation(array):
    meanArray = mean(array)
    for i in range(len(array)):
        array[i] = (array[i] - meanArray) * (array[i] - meanArray)
    variance = mean(array)
    return(math.sqrt(variance))

def calculateConfidenceIntervalls(results):

    accuracies = []
    for i in range(len(results)):
        accuracies.append(results[i][0])
    meanAccuracy = mean(accuracies)
    standardDeviation = standard_deviation(accuracies)
    
    print("Mean accuracy: " + str(meanAccuracy))
    print("Confidence: [" + str(meanAccuracy - 1.96 * standardDeviation) + ", " + str(meanAccuracy + 1.96 * standardDeviation) +"]")
