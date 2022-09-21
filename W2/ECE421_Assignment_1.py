
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

### Problem 1.1
### For the purpose of Problem 1.3, the return value of k_means algorithm includes the distortion, 
### which is not technically the requirement of 1.1.

data = load_breast_cancer()
         
def k_means(X, k):
    ### Random initialization of the centriods
    collection_set = [[] for i in range(k)]
    centroids_set = [[] for i in range(k)]
    for i in range(k):
        random_index = np.random.randint(0, 569)
        
        centroids_set[i] = X[random_index]
 
    
    ### Initialize end condition
    J_old = 0
    J_new = 0
    end = True
    ### Start optimization
    while end:
    ### Start classification
        for i in range(len(X)):
            d = 1000000 ### A random large number to start iteration.
            for m in range(k):
                if d > np.linalg.norm(np.array(X[i])-np.array(centroids_set[m])):
                    d = np.linalg.norm(np.array(X[i])-np.array(centroids_set[m]))
                    pointer = m
                    
            collection_set[pointer].append(X[i])
        # print(len(collection_set[0]),len(collection_set[1]))
    
        ### Now the raw data points are classified based on our original centroids
        ### We start to calculate and compare the distorsion 
        
        J_old = J_new ### Memorize the original distorsion value
        J_new = 0
        
        ### We sum up distorsion for all clusters
        for i in range(len(collection_set)):
            for j in range(len(collection_set[i])): 
                J_new += np.square((np.linalg.norm(np.array(collection_set[i][j])-np.array(centroids_set[i]))))
        
        # print("This is J_old: ", J_old)
        # print("This is J_new: ", J_new)
        
        if J_old != 0:
            if abs((J_old-J_new)/J_old) < 0.005: ### If the distorsion value stops changing rapidly
                # print("good")
                # print(abs((J_old-J_new)/J_old))
                end = False ### Time to end the loop
                
            else:
                # print(abs((J_old-J_new)/J_old))
                end = True
        
        ### Update the centroids by calculating mean
        for i in range(k):
            centroids_set[i] = np.mean(collection_set[i], axis=0)
            # print("This is new centroids: ", centroids_set[i])
        
    return centroids_set, collection_set, J_new


### Problem 1.2

# for i in range(2,8):
#     A, B, C = k_means(data.data, i)

# The two inputs I passed into k-mean algorithm is the expected value of k (number of clusters)
# as well as an array with 569 sub-arrays. Each sub-array contains 30 elements (dimensions). 



# Problem 1.3
res = []
for i in range(2,8):
    A, B, C = k_means(data.data, i)
    res.append(C)
    print("The distortion when i =", i, "is: ", C)
    
plt.plot([2, 3, 4, 5, 6, 7], res)
plt.show()


# Problem 1.4
# After run the code multiple times, I found that the inflection point
# usually occurs at k = 5. As result, I will choose k = 5 considering the accuracy of the result
# as well as the efficiency (run time) of the code.
