import nbimporter
import numpy as np
import skimage
import multiprocess
import threading
import queue
import os,time
import math
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from p1 import get_visual_words
from p2 import get_image_feature, distance_to_set
from util import get_num_CPU


def evaluate_recognition_system(num_workers=4):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    '''
    HINTS
    (1) You may wish to use multiprocessing to improve speed (NO Extra Points)
    (2) You may create helper function (in the same cell) to enable multiprocessing
    (3) Think Nearest Neighbor -> assign label using element closest in train set
    '''
    
    test_data = np.load("./data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    
    image_names = test_data['files']
    test_labels = test_data['labels']

    trained_features = trained_system['features']
    train_labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    SPM_layer_num = trained_system['SPM_layer_num']
    SPM_layer_num = int(SPM_layer_num)
    K = dictionary.shape[0]
    clf = MLPClassifier(random_state=1, max_iter=1000).fit(trained_features, train_labels)
    print("Trained features shape: ", trained_features.shape)
    
    # ----- TODO -----
    '''
    HINTS:
    1.> Think almost exactly similar to Q1.2.2
    2.> Create a list of arguments and use multiprocessing library
    3.> We can define a helper function which can take in the arguments (file_path, dictionary, SPM_layer_num,
        trained_features,...) as input and return (file_path, label, nearest neighbor index)
    4.> We can use python dictionary and file_path to have the output in correct order
    '''
    # YOUR CODE HERE
    lists_of_args = []
    num_test = image_names.shape[0]
    ordered_labels = np.array([], dtype = int)
    for i in range(num_test):
        print(i)
        full_image_name = './data/' + image_names[i]
        test_feat = get_image_feature(full_image_name, dictionary, SPM_layer_num, K)[1]
        prediction = clf.predict(test_feat.reshape(1,-1))
        pred_label = prediction[0]
        print(pred_label)
        ordered_labels = np.append(ordered_labels, pred_label)

    # raise NotImplementedError()
    '''
    HINTS:
    1.> Can use the file_name (path) to place the labels back in original order of input to multiprocessing
    '''
    
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    # print("Predicted labels shape: ", ordered_labels.shape)
    # print("Test Labels shape:", test_labels.shape)
    '''
    HINT:
    1.> Compute the confusion matrix (8x8)
    2.> Remember to save and upload the confusion matrix
    '''
    # YOUR CODE HERE
    test_labels = np.array(test_labels, dtype = int)
    conf_matrix = np.zeros([8,8])
    for i in range(len(test_labels)):
        conf_matrix[test_labels[i], ordered_labels[i]] = conf_matrix[test_labels[i], ordered_labels[i]] + 1

    accuracy = np.trace(conf_matrix)/np.sum(conf_matrix)

    # raise NotImplementedError()
    np.save("./conf_matrix_custom.npy",conf_matrix)
    return conf_matrix, accuracy


# NOTE: comment out the lines below before submitting to gradescope
# print(get_num_CPU())
conf, accuracy = evaluate_recognition_system(get_num_CPU())
# We expect the accuracy to be greater than 0.45
print("Accuracy:", accuracy)
print(conf)