class SVM_Result:
    
    def __init__(self, probability, distance):
        self.current = 0
        self.probability = probability
        self.distance = distance
        
    # Definitons to satisfy the iterable protocoll
        
    def __iter__(self):
        return self
        
    def __next__(self): 
        if self.current > 1:
            raise StopIteration
        elif self.current == 0:
            self.current += 1
            return self.probability
        else:
            self.current += 1
            return self.distance
            

class Points_Information:
    
    def __init__(self, results):
        
        # Invert the list to get the probabilites and the distances for each svm
        unzips = list(zip(*results))
        
        probabilites_for_each_svm = unzips[0]
        distances_for_each_svm = unzips[1]
        
        self.probabilites = list(zip(*probabilites_for_each_svm))
        self.distances = list(zip(*distances_for_each_svm))