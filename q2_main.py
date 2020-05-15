import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math
from knn import knn
from sklearn.metrics import mean_squared_error
from statistics import mean

def get_plot(x, y, x_label = '', y_label = '', title = 'plot'):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.show()


# Feature Scaling or Standardization
def normalization(matrix):
    mm_scaler = preprocessing.MinMaxScaler()
    matrix = mm_scaler.fit_transform(matrix)

    return matrix

''' 
input: get dataset and k (number of folds) and devide dataset to k fold
output: indices: a list containing end index of each fold 
        folded_data: a list containing tuple of each fold's train and test data '''
def kfold(data, k = 3):
    m = len(data)
    indices = []

    if k == 0:
        raise ValueError('k cannot be 0!!')
    if math.fmod(m, k) != 0:
        raise ValueError('Your data cannot be divided by k')
    else:
        fold_len = m/k
    prev_index = fold_len
    for i in range(k):
        indices.append(prev_index)
        prev_index = prev_index + fold_len

    # perform k_fold_cross_validation
    start_index = 0
    dataset = np.asarray(data)
    folded_data = []
    for i in range(0, k):
        end_index = int(indices[i])
        train_data = np.append(dataset[:start_index], dataset[end_index:], axis=0)
        test_data = dataset[start_index:end_index]
        folded_data.append((train_data,test_data))
        start_index = end_index
    return indices, folded_data

def knn_cross_validation(dataset, k, fold, regression=True, dist_type='manhattan'):
    fold_indices, folded_data = kfold(dataset, fold)
    fold_mse = []
    for i in range(fold):
        train_data = folded_data[i][0]
        test_data = folded_data[i][1]
        x_train = train_data[:, 0]
        y_train = train_data[:, 1]
        x_test = test_data[:, 0]
        y_test = test_data[:, 1]
        y_pred = knn(k, x_train, y_train, x_test, dist_type, regression)
        MSE = mean_squared_error(y_test, y_pred)
        fold_mse.append(MSE)
    avg_mse = mean(fold_mse)
    return avg_mse

if __name__ == '__main__':
    ################## loading data and preprocessing ###################
    dataset = open("./dataset/regression.txt", encoding="utf8").read()
    dataset = np.matrix(dataset).reshape(240,2)
    # normalization
    dataset = normalization(dataset)

    # plt.plot(dataset[:,0],dataset[:,1])
    # plt.show()
    # plt.savefig("dataset after normalization")
    # plt.clf()
    ###################### end of preprocessing #########################

    # #################### regression using k fold cross validation ####################
    # _____________________________question 2  part A ________________________________
    # fold_list = [3,5,8,10]
    # k_list = np.arange(1,16)
    # plt_lbl = []
    #
    # for fold in fold_list:
    #     MSE_history = []
    #
    #     for k in k_list:
    #         MSE = knn_cross_validation(dataset, k, fold, True, 'manhattan')
    #         MSE_history.append(MSE)
    #         print("fold:", fold,"k:", k, "MSE:", MSE)
    #     get_plot(k_list, MSE_history, 'k', 'MSE', 'find best fold and k value')
    #     plt_lbl.append('fold = ' + str(fold))
    # plt.legend(plt_lbl)
    # plt.savefig('choose best parameters')
    # # plt.show()
    # plt.clf()
    # _____________________________question 2  part B ________________________________
    fold = 5
    k = 3

    fold_indices, folded_data = kfold(dataset, fold)
    train_data = folded_data[2][0]
    test_data = folded_data[2][1]
    y_pred = knn(k, train_data[:,0], train_data[:,1], test_data[:,0], 'manhattan', True)
    get_plot(train_data[:,0], train_data[:,1])
    get_plot(test_data[:,0],test_data[:,1],'x','y','train and test data for k='+str(k) + ' and fold='+str(fold))
    # get_plot(test_data[:,0], y_pred,'x','y','train and test data for best k')
    plt.savefig('q2_partB')
    # print(y_pred)
