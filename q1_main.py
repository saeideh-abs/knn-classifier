from sklearn.model_selection import train_test_split
import numpy as np
from knn import knn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

def evaluation(test_labels, predicted_test_labels):
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    print("accuracy:", accuracy)
    conf_mat = confusion_matrix(test_labels, predicted_test_labels)
    tn, fp, fn, tp = confusion_matrix(test_labels, predicted_test_labels).ravel()
    print("confusion matrix: \n ", conf_mat)
    print("(tn, fp, fn, tp)")
    print((tn, fp, fn, tp))
    return (accuracy, conf_mat, (tn, fp, fn, tp))

# Feature Scaling or Standardization
def normalization(matrix):
    mm_scaler = preprocessing.MinMaxScaler()
    matrix = mm_scaler.fit_transform(matrix)

    return matrix

if __name__ == '__main__':

    ################## loading data and preprocessing ###################
    data = open("./dataset/processed.cleveland.data", encoding="utf8").read().split('\n')

    heart_disease_dataset = []
    for index, line in enumerate(data):
        heart_disease_dataset.append([])
        heart_disease_dataset[index] = [n for n in line.split(',')]
    heart_disease_dataset = np.array(heart_disease_dataset)

    #-------------- choose mean value for features with unkown value -------------------#
    rows_number, cols_number = np.where(heart_disease_dataset == '?')
    reduced_matrix = heart_disease_dataset
    # remove entire row of unkown values from reduced_matrix
    reduced_matrix = np.delete(reduced_matrix, rows_number, 0)
    # convert string values(after using split) to float
    reduced_matrix = reduced_matrix.astype(np.float)

    # put column mean for unkown values
    for index, j in enumerate(cols_number):
        mean = np.mean(reduced_matrix[:, j])
        heart_disease_dataset[rows_number[index]][j] = mean
    # convert string values(after using split) to float
    heart_disease_dataset = heart_disease_dataset.astype(np.float)

    train_data, test_data = train_test_split(heart_disease_dataset, test_size=0.3)

    train_labels = train_data[:, -1] #get last column
    train_data = np.delete(train_data, -1, 1) #drop last column (labls column)
    test_labels = test_data[:, -1]  # get last column
    test_data = np.delete(test_data, -1, 1)  # drop last column (labls column)

    # set two classes for labels. just 0 and 1.
    for index, label in enumerate(train_labels):
        if label>0 :
            train_labels[index] = 1
    for index, label in enumerate(test_labels):
        if label>0 :
            test_labels[index] = 1
    ######################## end of preprocessing part ############################

    # _______________________ question 1 - part A _______________________
    print("question 1- part A:")
    k_list = [1,2,3,4,5,6,7,10,15]
    for k in k_list:
        predicted_test_labels = knn(k, train_data, train_labels, test_data, 'euclidean')
        print("***** k = ", k, "*****")
        # evaluation part
        evaluation(test_labels, predicted_test_labels)

    # _______________________ question 1 - part B _______________________
    # normalization
    normal_train_data = normalization(train_data)
    normal_test_data = normalization(test_data)
    print("********** results after using normalization ***********")
    print("********** results when using euclidean distance ***********")
    k = 4
    predicted_test_labels = knn(k, normal_train_data, train_labels, normal_test_data, 'euclidean')
    evaluation(test_labels, predicted_test_labels)

    # _______________________ question 1 - part C _______________________
    print("********** results when using manhattan distance ***********")
    predicted_test_labels = knn(k, normal_train_data, train_labels, normal_test_data, 'manhattan')
    evaluation(test_labels, predicted_test_labels)

    print("********** results when using chebyshev distance ***********")
    predicted_test_labels = knn(k, normal_train_data, train_labels, normal_test_data, 'chebyshev')
    evaluation(test_labels, predicted_test_labels)