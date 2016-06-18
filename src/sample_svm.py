import random
from sklearn import svm
from models import *
import matplotlib.pyplot as plt
from numpy import *
import datetime

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
    
    #training_data, prediction_data = input_data
    
    training_data = input_data.training_data
    prediction_data = input_data.prediction_data
    kernel = input_data.kernel
    
    # Set seed for each Thread new as otherwise each process starts with same Seed -> ugly
    SEED = datetime.datetime.now().time().microsecond
    random.seed(SEED)
    
    y, X = random_sample_with_replacement_of_dataset(training_data, len(training_data[0]))
    
    clf = svm.SVC(probability = True, kernel = kernel, C = input_data.C)
    fit = clf.fit(X, y)  
    
    
    ### Berechnugn linine
    if(kernel == 'linear'):
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = linspace(0, 1)
        yy = a * xx - (clf.intercept_[0]) / w[1]
    
    
    distance_to_hyperplane = clf.decision_function(prediction_data[1])
    probabilities = clf.predict_proba(prediction_data[1])
    probabilities = list(zip(*probabilities))
    probabilities = probabilities[1]
    
    
    result = SVM_Result(probabilities, distance_to_hyperplane, yy, clf.n_support_, 0)
    
    return(result)
    
    
def do_svm(input_data):
    
    training_data = input_data.training_data
    prediction_data = input_data.prediction_data
    kernel = input_data.kernel
    
    y, X = training_data
    
    clf = svm.SVC(probability = True, kernel = kernel, C = input_data.C)
    fit = clf.fit(X, y)  
    
    
    ### Berechnugn linine
    if(kernel == 'linear'):
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = linspace(0, 1)
        yy = a * xx - (clf.intercept_[0]) / w[1]
    
    
    distance_to_hyperplane = clf.decision_function(prediction_data[1])
    probabilities = clf.predict_proba(prediction_data[1])
    probabilities = list(zip(*probabilities))
    probabilities = probabilities[1]
    
    
    score = clf.score(prediction_data[1],prediction_data[0])
    
    #print(distance_to_hyperplane)
    #print(probabilities)
    
    result = SVM_Result(probabilities, distance_to_hyperplane, yy, clf.n_support_, score)
    
    return(result)
    
    
    