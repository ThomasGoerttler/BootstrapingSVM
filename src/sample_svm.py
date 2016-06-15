import random
from sklearn import svm
from models import *

def random_sample_with_replacement(population, sample_size):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [_int(_random() * n) for i in range(sample_size)]

def random_sample_with_replacement_of_dataset(data, sample_size):
        y, x = data
        sorted_list = sorted(random_sample_with_replacement(range(len(y)), sample_size))
        y = [y[i] for i in sorted_list]
        x = [x[i] for i in sorted_list]
        return(y, x)


def single_sample_and_svm(input_data):
    "New version of using skilearn"
    
    training_data, prediction_data = input_data
    y, X = random_sample_with_replacement_of_dataset(training_data, len(training_data[0]))
    clf = svm.SVC(probability = True, kernel='rbf')
    fit = clf.fit(X, y)  
    
    distance_to_hyperplane = clf.decision_function(prediction_data[1])
    probabilities = clf.predict_proba(prediction_data[1])
    probabilities = list(zip(*probabilities))
    probabilities = probabilities[1]
    
    print(distance_to_hyperplane)
    print(probabilities)
    
    result = SVM_Result(probabilities,distance_to_hyperplane)
    
    return(result)
    
    
    
    