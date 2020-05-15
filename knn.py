import numpy as np
from collections import Counter
from statistics import mean

''' multiclass k nearest neighbor with ability to use for regression
    set regression = True to use regression and regression = False for classification
    the parameter distance_type get 3 value:
    1-euclidean 2-manhattan 3-chebyshev
    then the distance between the two points will be calculated according to desired distance_type
'''
def knn(k, x_train, y_train, x_test, distance_type = 'manhattan', regression = False):
    y_test = []
    calculate_dist = switch_dist_type(distance_type)

    for test_vec in x_test:
        test_sample_distance = [] # distances from test sample to all train samples
        k_nearest_neighbor = [] # k nearest neighbor to test sample
        for train_index, train_vec in enumerate(x_train):
            test_sample_distance.append(calculate_dist(test_vec, train_vec))
        test_sample_distance = np.array(test_sample_distance)
        max_index = np.argmax(test_sample_distance)
        for i in range(k):
            min_index = np.argmin(test_sample_distance)
            k_nearest_neighbor.append(y_train[min_index]) #get label/value of most nearest train sample to test sample
            test_sample_distance[min_index] = test_sample_distance[max_index] #replace minimum value with maximum value in order to prevent reselection
        if regression == False:
            # voting between labels / choose the most common label
            count = Counter(k_nearest_neighbor)
            label = count.most_common(1)
            y_test.append(label[0][0])
        else:
            mean_value = mean(k_nearest_neighbor)
            y_test.append(mean_value)

    return y_test

# switch-case for selecting distance calculation type
def switch_dist_type(type):
    switcher = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'chebyshev': chebyshev_distance,
    }
    return switcher.get(type, manhattan_distance)

#calculate euclidean distance between 2 vectors
def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.size != vec2.size:
        raise ValueError('function euclidean_distance: length of two vectors should be equal!')
    else:
        residual  = np.subtract(vec1, vec2)
        square = np.power(residual, 2)
        summation = np.sum(square)
        distance = np.sqrt(summation)
    return distance

def manhattan_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.size != vec2.size:
        raise ValueError('function manhattan_distance: length of two vectors should be equal!')
    else:
        residual = np.subtract(vec1, vec2)
        absolute = np.absolute(residual)
        summation = np.sum(absolute)
    return summation

def chebyshev_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.size != vec2.size:
        raise ValueError('function chebyshev_distance: length of two vectors should be equal!')
    else:
        residual = np.subtract(vec1, vec2)
        absolute = np.absolute(residual)
        max_index = np.argmax(absolute)
    return absolute[max_index]