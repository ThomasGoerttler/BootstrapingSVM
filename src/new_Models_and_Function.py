from functions import * 

class Bootstrap_Result:

    def __init__(self, svm, accuracy, classification, var_probability, var_distance, n_support):
        self.svm = svm
        self.accuracy = accuracy
        self.classification = classification
        self.var_probability = var_probability
        self.var_distance = var_distance
        self.n_support = n_support
        
    def view(self):
        print()
        print("Result of Bootstrap")
        print()
        print(self.svm)
        print("Accuaray:", self.accuracy)
        print("Classification:", self.classification)
        print("Variance in Probabilty:", self.var_probability)
        print("Variance in distance:", self.var_distance)
        print("Number of Suppotvectors:",self.n_support)
        
    # Definitons to satisfy the iterable protocoll
        
    def __iter__(self):
        return self
        
    def __next__(self): 
        if self.current > 5:
            raise StopIteration
        elif self.current == 0:
            self.current += 1
            return self.svm
        elif self.current == 1:
            self.current += 1
            return self.accuracy
        elif self.current == 2:
            self.current += 1
            return self.classification
        elif self.current == 3:
            self.current += 1
            return self.var_probability
        elif self.current == 4:
            self.current += 1
            return self.var_distance
        else:
            self.current += 1
            return self.n_support
    
def do_Boot(trainings_data, prediction_data, kernel, C, gamma, degree, processes, replications):
    result = do_Bootstrap(trainings_data, prediction_data, kernel, C, gamma, degree, processes, replications)
    res = Bootstrap_Result(result[0], result[1], result[2], result[3], result[4], result[0].n_support_ )
    return(res)
    