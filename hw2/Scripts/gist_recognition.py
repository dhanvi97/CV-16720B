# Do Not Modify
import nbimporter
from util import display_filter_responses, save_wordmap, get_num_CPU
import numpy as np
import multiprocess
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import random
import cv2
import math

from skimage import io
#-------------------------------------------------------------------------

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''
    
    
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(image.shape == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]

    image = skimage.color.rgb2lab(image)

    H = image.shape[0] 
    W = image.shape[1]
    filter_responses = np.array([])
    '''
    HINTS: 
    1.> Iterate through the scales (5) which can be 1, 2, 4, 8, 8$sqrt{2}$
    2.> use scipy.ndimage.gaussian_* to create filters
    3.> Iterate over each of the three channels independently
    4.> stack the filters together to (H, W,3F) dim
    '''
    # ----- TODO -----
    
    # YOUR CODE HERE
    scales = np.array([0.05,0.1,0.4,0.8])
    ch1 = image[:,:,0]
    ch2 = image[:,:,1]
    ch3 = image[:,:,2]
    gamma = 1 
    for sigma in scales:
        hor_ch1_r, hor_ch1_i = skimage.filters.gabor(ch1, sigma, theta=0, mode ='constant')
        hor_ch2_r, hor_ch2_i = skimage.filters.gabor(ch2, sigma, theta=0, mode ='constant')
        hor_ch3_r, hor_ch3_i = skimage.filters.gabor(ch3, sigma, theta=0, mode ='constant')
        # hor_ch1 = np.hypot(hor_ch1_r, hor_ch1_i); hor_ch2 = np.hypot(hor_ch2_r, hor_ch2_i); hor_ch3 = np.hypot(hor_ch3_r, hor_ch3_i)
        gstack = np.dstack([hor_ch1_r, hor_ch2_r, hor_ch3_r])
        ver_ch1_r, ver_ch1_i = skimage.filters.gabor(ch1, sigma, theta=np.pi/2,  mode ='constant')
        ver_ch2_r, ver_ch2_i = skimage.filters.gabor(ch2, sigma, theta=np.pi/2,  mode ='constant')
        ver_ch3_r, ver_ch3_i = skimage.filters.gabor(ch3, sigma, theta=np.pi/2,  mode ='constant')
        # ver_ch1 = np.hypot(ver_ch1_r, ver_ch1_i); ver_ch2 = np.hypot(ver_ch2_r, hor_ch2_i); ver_ch3 = np.hypot(ver_ch3_r, ver_ch3_i)
        lstack = np.dstack([ver_ch1_r, ver_ch2_r, ver_ch3_r])
        thirty_ch1_r, thirty_ch1_i = skimage.filters.gabor(ch1, sigma, theta=np.pi/6,  mode ='constant')
        thirty_ch2_r, thirty_ch2_i = skimage.filters.gabor(ch2, sigma, theta=np.pi/6,  mode ='constant')
        thirty_ch3_r, thirty_ch3_i = skimage.filters.gabor(ch3, sigma, theta=np.pi/6,  mode ='constant')
        # thirty_ch1 = np.hypot(thirty_ch1_r, thirty_ch1_i); thirty_ch2 = np.hypot(thirty_ch2_r, thirty_ch2_i); thirty_ch3 = np.hypot(thirty_ch3_r, thirty_ch3_i)
        dogxstack = np.dstack([thirty_ch1_r, thirty_ch2_r, thirty_ch3_r])
        sixty_ch1_r, sixty_ch1_i = skimage.filters.gabor(ch1, sigma, theta=np.pi/3,  mode ='constant')
        sixty_ch2_r, sixty_ch2_i = skimage.filters.gabor(ch2, sigma, theta=np.pi/3,  mode ='constant')
        sixty_ch3_r, sixty_ch3_i = skimage.filters.gabor(ch3, sigma, theta=np.pi/3,  mode ='constant')
        # sixty_ch1 = np.hypot(sixty_ch1_r, sixty_ch1_i); sixty_ch2 = np.hypot(sixty_ch2_r, sixty_ch2_i); sixty_ch3 = np.hypot(sixty_ch3_r, sixty_ch3_i)
        dogystack = np.dstack([sixty_ch1_r, sixty_ch2_r, sixty_ch3_r])
        ff_ch1_r, ff_ch1_i = skimage.filters.gabor(ch1, sigma, theta=np.pi/4,  mode ='constant')
        ff_ch2_r, ff_ch2_i = skimage.filters.gabor(ch2, sigma, theta=np.pi/4,  mode ='constant')
        ff_ch3_r, ff_ch3_i = skimage.filters.gabor(ch3, sigma, theta=np.pi/4,  mode ='constant')
        # ff_ch1 = np.hypot(ff_ch1_r, ff_ch1_i); ff_ch2 = np.hypot(ff_ch2_r, ff_ch2_i); ff_ch3 = np.hypot(ff_ch3_r, ff_ch3_i)
        ffstack = np.dstack([ff_ch1_r, ff_ch2_r, ff_ch3_r])
        of_ch1_r, of_ch1_i = skimage.filters.gabor(ch1, sigma, theta=np.pi/12, mode ='constant')
        of_ch2_r, of_ch2_i = skimage.filters.gabor(ch2, sigma, theta=np.pi/12, mode ='constant')
        of_ch3_r, of_ch3_i = skimage.filters.gabor(ch3, sigma, theta=np.pi/12, mode ='constant')
        # of_ch1 = np.hypot(of_ch1_r, of_ch1_i); of_ch2 = np.hypot(of_ch2_r, of_ch2_i); of_ch3 = np.hypot(of_ch3_r, of_ch3_i)
        ofstack = np.dstack([of_ch1_r, of_ch2_r, of_ch3_r])
        sf_ch1_r, sf_ch1_i = skimage.filters.gabor(ch1, sigma, theta=5*np.pi/12, mode ='constant')
        sf_ch2_r, sf_ch2_i = skimage.filters.gabor(ch2, sigma, theta=5*np.pi/12, mode ='constant')
        sf_ch3_r, sf_ch3_i = skimage.filters.gabor(ch3, sigma, theta=5*np.pi/12, mode ='constant')
        # sf_ch1 = np.hypot(sf_ch1_r, sf_ch1_i); sf_ch2 = np.hypot(sf_ch2_r, sf_ch2_i); sf_ch3 = np.hypot(sf_ch3_r, sf_ch3_i)
        sfstack = np.dstack([sf_ch1_r, sf_ch2_r, sf_ch3_r])
        filter_stack = np.dstack([gstack, ofstack, dogxstack, ffstack, dogystack, sfstack, lstack])
        filter_responses = np.dstack([filter_responses, filter_stack]) if filter_responses.size else filter_stack
        
    return filter_responses

    
def gist_per_image(image):
    filter_responses = extract_filter_responses(image)
    h,w,f = filter_responses.shape
    gist_descriptor = np.array([])
    for i in range(4):
        for j in range(4):
            patch = filter_responses[4*i:4*(i+1), 4*j:4*(j+1), :]
            mean_feat = np.average(patch, axis = (0,1))
            gist_descriptor = np.append(gist_descriptor, mean_feat)
    return gist_descriptor



def compute_dictionary_one_image(args):
    '''
    Extracts samples of the dictionary entries from an image. Use the the 
    harris corner detector implmented from previous question to extract 
    the point of interests. This should be a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''
    
    # ----- TODO -----
    '''
    HINTS:
    1.> Create a tmp dir to store intermediate results.
    2.> Read the image from image_path using skimage
    3.> extract filter responses and points of interest
    4.> store the response of filters at points of interest 
    '''
    i, image_path = args
    print('Train image - ' + str(i))
    image = io.imread(image_path)
    image = image.astype('float')/255

    feat = gist_per_image(image)
    
    return feat



    # YOUR CODE HERE

def train_gist_model(num_workers=4):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    train_data = np.load("./data/train_data.npz")
    # ----- TODO -----
    list_of_args = []
    
    '''
    Can change these values for experiments, however please submit the dictionary.npy with these values
    alpha=150 and n_clusters = 200
    '''

    image_names = train_data['files']
    labels = train_data['labels']
    num_images = image_names.shape[0]

    for i in range(num_images):
        full_image_name = './data/' + image_names[i]
        list_of_args.append([i, full_image_name])
    
    
    with multiprocess.Pool(num_workers) as p:
        output = p.map(compute_dictionary_one_image, list_of_args)
    
    '''

    HINTS:
    
    1.> Use multiprocessing for parallel processing of elements
    2.> Next, load the tmp files and stack the responses stored as npy
    '''
    # YOUR CODE HERE
    # raise NotImplementedError()
    # filter_responses = np.concatenate(filter_responses, axis=0)
    
    '''
    HINTS:
    1.> use sklearn.cluster.KMeans for clustersing
    2.> dictionary will be the cluster_centers_
    '''
    # YOUR CODE HERE
    ordered_features = np.array(output)
    labels = np.array(labels)

    # raise NotImplementedError()
    np.savez('trained_model_gist.npz', features=ordered_features, labels=labels)

    
# NOTE: comment out the lines below before submitting to gradescope

def helper_func(args):
    i, file_path, trained_features, train_labels = args
    print('Test image - ' + str(i))
    image = io.imread(file_path).astype('float')/255
    test_feat = gist_per_image(image)
    idx = np.argmin(np.sum(np.square(trained_features - test_feat), axis=1))
    pred_label = train_labels[idx]

    return pred_label

def eval_model(num_workers=16):
    test_data = np.load('./data/test_data.npz')

    trained_systems = np.load('trained_model_gist.npz')
    test_labels = test_data['labels']
    image_names = test_data['files']
    test_num = image_names.shape[0]

    trained_features = trained_systems['features']
    train_labels = trained_systems['labels']

    arg_list = []

    for i in range(test_num):
        full_image_name = './data/' + image_names[i]
        arg_list.append([i, full_image_name, trained_features, train_labels])

    with multiprocess.Pool(num_workers) as p:
        output = p.map(helper_func, arg_list)

    ordered_labels = np.array(output)
    test_labels = np.array(test_labels)

    conf_matrix = np.zeros([8,8])
    for i in range(len(test_labels)):
        conf_matrix[test_labels[i], ordered_labels[i]] = conf_matrix[test_labels[i], ordered_labels[i]] + 1

    accuracy = np.trace(conf_matrix)/np.sum(conf_matrix)

    np.save("./conf_matrix_gist.npy",conf_matrix)
    return conf_matrix, accuracy








if __name__=='__main__':
    # dictionary = np.load('dictionary.npy')
    # train_data = np.load('./data/train_data.npz')
    # image_names = train_data['files']
    # idx = np.random.randint(0, len(image_names)-1)
    # im_path = './data/' + image_names[idx]
    # print(im_path)
    
    im_path = './data/aquarium/sun_aydaknxraiwghvmi.jpg'
    image = io.imread(im_path)
    image = image.astype('float')/255

    # print(image.shape)
    filter_responses = extract_filter_responses(image)
    print(filter_responses.shape)
    display_filter_responses(filter_responses)
    feature = gist_per_image(image)
    # print(feature.shape)
    # compute_dictionary(16)
    # conf_matrix, accuracy = eval_model(16)
    # print(conf_matrix)
    # print('Accuracy: ' + str(accuracy))
    
    
    # compute_dictionary(16)
