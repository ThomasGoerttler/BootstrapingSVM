import random
from liblinearutil import *

def getRandomSampleWithReplacement(data, sample_size):
        y, x = data
        sorted_list = sorted(random.sample(xrange(len(y)), sample_size))
        y = [y[i] for i in sorted_list]
        x = [x[i] for i in sorted_list]
        return(y, x)

def bootstrapping(data):
    
    y, x = getRandomSampleWithReplacement(data, 251)
    m = train(y[:200], x[:200], '-c 4')
    p_label, p_acc, p_val = predict(y[200:], x[200:], m)
    
    return(p_acc)
