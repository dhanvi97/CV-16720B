import nbimporter
import numpy as np
import skimage
import multiprocess
import threading
import queue
import os,time
import math
import matplotlib.pyplot as plt
from p1 import get_visual_words
from util import get_num_CPU, save_wordmap

def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    '''
    HINTS:
    (1) We can use np.histogram with flattened wordmap
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    flattened_wordmap = wordmap.flatten()
    bins_equi = np.arange(0, dict_size+1, 1)
    hist_data = np.histogram(flattened_wordmap, bins=bins_equi, density=True)
    hist = hist_data[0]
#     raise NotImplementedError()
    return hist

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    '''
    HINTS:
    (1) Take care of Weights 
    (2) Try to build the pyramid in Bottom Up Manner
    (3) the output array should first contain the histogram for Level 0 (top most level) , followed by Level 1, and then Level 2.
    '''
    # ----- TODO -----
    h, w = wordmap.shape
    L = layer_num - 1
    patch_width = math.floor(w / (2**L))
    patch_height = math.floor(h / (2**L))
    
    '''
    HINTS:
    1.> create an array of size (dict_size, (4**(L + 1) -1)/3) )
    2.> pre-compute the starts, ends and weights for the SPM layers L 
    i = 0 0 + 1 + 4 + 5
    i = 1 2 + 3 + 6 + 7
    i = 2 8 + 9 + 12 + 13
    i = 3 10 + 11 + 14 + 15

    i = 0 0 + 1 + 16 + 17
    i = 1 2 + 3 + 18 + 19
    i = 2  + 4 + 20 + 21
    .
    .
    .
    i = 7 9 + 10 

    '''
    # YOUR CODE HERE
    num_feat = math.floor((4**(L+1)-1)/3)
    histogram_arr = np.zeros([dict_size, num_feat])
    row_cuts = np.arange(0, h, patch_height)
    col_cuts = np.arange(0, w, patch_width)
    bins_equi = np.arange(0, dict_size + 1, 1)

    weights = np.array([])
    for l in range(layer_num):
        if l == 0:
            arr = np.array([2**(-L)])
        else:
            arr = 2**(l-L-1)*np.ones(4**l)
        weights = np.concatenate([weights, arr])

    for i in range(len(row_cuts)):
        for j in range(len(col_cuts)):
            patch = wordmap[row_cuts[i]:row_cuts[i] + patch_height+1, col_cuts[j]:col_cuts[j] + patch_width + 1]
            # patch_count = patch.shape[0]*patch.shape[1]

            patch_hist = np.histogram(patch, bins=bins_equi, density = False)
            patch_hist = patch_hist[0]
            histogram_arr[:, -4**L + len(col_cuts)*j + i] = patch_hist       
#     raise NotImplementedError()
    '''
    HINTS:
    1.> Loop over the layers from L to 0
    2.> Handle the base case (Layer L) separately and then build over that
    3.> Normalize each histogram separately and also normalize the final histogram
    '''
    # YOUR CODE HERE
    l = L-1
    while l >0 :
        histogram_lower = histogram_arr[:,math.floor((4**(l+1)-1)/3):math.floor((4**(l+2)-1)/3) + 1]
        for i in range(2**l):
            for j in range(2**l):
                k_i = 2*i; k_j = 2*j
                histogram_arr[:, 2**l*j + i] = histogram_lower[:, 2**(l+1)*k_j + k_i] + histogram_lower[:, 2**(l+1)*(k_j+1) + k_i] + histogram_lower[:, 2**(l+1)*(k_j) + k_i+1] +histogram_lower[:, 2**(l+1)*(k_j+1) + k_i+1]

        # print(histogram_lower.shape)
        l= l -1
    histogram_arr[:,0] = np.histogram(wordmap, bins=bins_equi, density = False)[0]
    total = np.sum(histogram_arr)
    histogram_arr = histogram_arr/total
    histogram_arr = np.multiply(histogram_arr, weights)
    hist_all = histogram_arr.flatten()
    # print(hist_all.shape)





    
#     raise NotImplementedError()
    return hist_all

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    '''
    HINTS:
    (1) Consider A = [0.1,0.4,0.5] and B = [[0.2,0.3,0.5],[0.8,0.1,0.1]] then \
        similarity between element A and set B could be represented as [[0.1,0.3,0.5],[0.1,0.1,0.1]]   
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    
#     raise NotImplementedError()
    min_mat = np.minimum(word_hist, histograms)
    sim = np.sum(min_mat, axis = 1)
    return sim

def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255
    
    wordmap = get_visual_words(image,dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    
#     raise NotImplementedError()
    return [file_path, feature]

def build_recognition_system(num_workers):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("./data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    # YOUR CODE HERE
    images_names = train_data['files']
    num_images = images_names.shape[0]
    labels = train_data['labels']
    # print(labels.shape)

    lists_of_args = []

    SPM_layer_num = 3
    K = dictionary.shape[0]

    for i in range(num_images):
        full_image_name = './data/' + images_names[i]
        lists_of_args.append([full_image_name, dictionary, SPM_layer_num, K])

    with multiprocess.Pool(num_workers) as p:
        output = p.starmap(get_image_feature, lists_of_args)

    ordered_features = [x[1] for x in output]



    # raise NotImplementedError()
    np.savez('trained_system.npz', features=ordered_features,
                                    labels=labels,
                                    dictionary=dictionary,
                                    SPM_layer_num=SPM_layer_num)


# NOTE: comment out the lines below before submitting to gradescope
build_recognition_system(16)

# train_data = np.load("./data/train_data.npz")
# image_names = train_data['files']
# dictionary = np.load('dictionary.npy')
# dict_size = dictionary.shape[0]
# bins_equi = np.arange(0, dict_size+1, 1)

# for t in range(10,15):
#     idx = np.random.randint(0,len(image_names)-1)
#     test_image_1 = "./data/" + image_names[t]
#     image = skimage.io.imread(test_image_1)
#     image = image.astype('float')/255
#     wordmap = get_visual_words(image, dictionary)
#     save_wordmap(wordmap, 'wordmap' + str(t))




# histogram = get_feature_from_wordmap_SPM(wordmap,3, dict_size)