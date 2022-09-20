from unittest import skip
from sklearn.datasets import load_breast_cancer
import numpy as np
data = load_breast_cancer()
# Test 
# print(data.target[[1,2,3,4,5,6,7,8,9,20,21]]) 569
# print(type(data.target)) : numpy.ndarray

X, Y = load_breast_cancer(return_X_y=True)


### Helper function: calculate distance between two points
def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
### Test on helper function:
# print(distance([6,7],[3,3]))

### Helper function: Cost Function
def cost_function(a,b):
    res = 0
    for i in range(len(a)):
        res += distance(a[i],b)**2
        
    res = res/len(a)
    #print(res)
    return res



def k_means(X_set, Y_set, k):
    
    ### Random initialization of the centriods
    collection_set = [[] for i in range(k)]
    centroids_set = [[None for i in range(2)] for i in range(k)]
    for i in range(k):
    # Pick raw data points as the initial centroids
        centroids_set[i][0] = X[i][0]
        centroids_set[i][1] = Y[i]
    # print(centroids_set)
    ### Start classification
    for i in range(len(X_set)):
        for j in range(len(X_set[i])):
           a = [X_set[i][j],Y_set[i]] 
           d = 100000 ### A random large number to start iteration.
           for m in range(k):
                b = centroids_set[m]
                if d > distance(a,b):
                    #print(d)
                    d = distance(a,b)
                    pointer = m
                    
           collection_set[pointer].append(a)
    # print(len(collection_set[1]))
    
    ### Now the raw data points are classified based on our original centroids
    cost_function(collection_set[1], centroids_set[1])
    ### Next we calculate the mean
    
    return None

k_means(X, Y, 2)