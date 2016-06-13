import random
from sklearn import svm

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


def single_sample_and_svm(training_data):
    "New version of using skilearn"
    
    prediction_data = training_data
    
    y, X = random_sample_with_replacement_of_dataset(training_data, len(training_data[0]))

    clf = svm.SVC()
    fit = clf.fit(X, y)  
    
    y_pred, X_pred = prediction_data
    
    probabilities = clf.decision_function(X_pred)
    
    print(probabilities)
    
    return(probabilities)
    
    
    
    