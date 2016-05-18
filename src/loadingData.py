from liblinearutil import *

def loadData(filename):
    return(svm_read_problem('./' + filename))