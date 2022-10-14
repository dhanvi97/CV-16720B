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

from skimage import io
#-------------------------------------------------------------------------


def plot_harris_points(image,points):
    fig = plt.figure(1)
    for x,y in zip(points[0],points[1]):
        plt.plot(y,x,marker='v')
    plt.imshow(image)
    plt.show()

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
    scales = np.array([1,2,4,8,8*np.sqrt(2), 11])
    ch1 = image[:,:,0]
    ch2 = image[:,:,1]
    ch3 = image[:,:,2]
    for sigma in scales:
        gauss_ch1 = scipy.ndimage.gaussian_filter(ch1, sigma, mode ='constant')
        gauss_ch2 = scipy.ndimage.gaussian_filter(ch2, sigma, mode ='constant')
        gauss_ch3 = scipy.ndimage.gaussian_filter(ch3, sigma, mode ='constant')
        gstack = np.dstack([gauss_ch1, gauss_ch2, gauss_ch3])
        laplace_ch1 = scipy.ndimage.gaussian_laplace(ch1, sigma, mode = 'constant')
        laplace_ch2 = scipy.ndimage.gaussian_laplace(ch2, sigma, mode = 'constant')
        laplace_ch3 = scipy.ndimage.gaussian_laplace(ch3, sigma, mode = 'constant')
        lstack = np.dstack([laplace_ch1, laplace_ch2, laplace_ch3])
        dog_x_ch1 = scipy.ndimage.gaussian_filter(ch1, sigma, order = (0,1), mode='constant')
        dog_x_ch2 = scipy.ndimage.gaussian_filter(ch2, sigma, order = (0,1), mode='constant')
        dog_x_ch3 = scipy.ndimage.gaussian_filter(ch3, sigma, order = (0,1), mode='constant')
        dogxstack = np.dstack([dog_x_ch1, dog_x_ch2, dog_x_ch3])
        dog_y_ch1 = scipy.ndimage.gaussian_filter(ch1, sigma, order = (1,0), mode='constant')
        dog_y_ch2 = scipy.ndimage.gaussian_filter(ch2, sigma, order = (1,0), mode='constant')
        dog_y_ch3 = scipy.ndimage.gaussian_filter(ch3, sigma, order = (1,0), mode='constant')
        dogystack = np.dstack([dog_y_ch1, dog_y_ch2, dog_y_ch3])
        filter_stack = np.dstack([gstack, lstack, dogxstack, dogystack])
        filter_responses = np.dstack([filter_responses, filter_stack]) if filter_responses.size else filter_stack
        
    return filter_responses

def unittest_extract_filter_response():
    train_data = np.load("./data/train_data.npz")
    image_names = train_data['files']
    for t in range(0,5):
        idx = np.random.randint(0,len(image_names)-1)
        test_image_1 = "./data/" + image_names[idx]
        # test_image_1 = "./data/aquarium/sun_axahqdyqpausckwh.jpg"
        image = io.imread(test_image_1)
        image = image.astype('float')/255

        h,w = image.shape[0],image.shape[1]
        num_channels = 60
        
        std_filter_response = extract_filter_responses(image)
        try:
            assert std_filter_response.shape[0] == h
        except:
            raise AssertionError('Test Case {}: failed  Wrong Height'.format(idx+1))
        try:
            assert std_filter_response.shape[1] == w
        except:
            raise AssertionError('Test Case {}: failed  Wrong Width'.format(idx+1))
        try:
            assert std_filter_response.shape[2] == num_channels
        except:
            raise AssertionError('Test Case {}: failed  Wrong Channels'.format(idx+1))
        
        
        image = image[:,:,0]
        try:
            std_filter_response = extract_filter_responses(image)
        except:
            raise AssertionError('Test Case {}: failed  Cannot Handle Different Input Format'.format(idx+1))
    
def get_harris_corners(image, alpha, k = 0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor 

    [output]
    * points_of_interest: numpy.ndarray of shape (2, alpha) that contains interest points
    '''
    
    '''
    HINTS:
    (1) Visualize and Compare results with cv2.cornerHarris() for debug (DO NOT SUBMIT cv2's implementation)
    '''
    # ----- TODO -----
    
    ######### Actual Harris #########
    from skimage.color import rgb2gray
    from scipy import ndimage


    bw_img = rgb2gray(image)
    # bw_img = bw_img - np.mean(bw_img)
    
    '''
    HINTS:
    1.> For derivative images we can use cv2.Sobel filter of 3x3 kernel size
    2.> For double derivate (e.g. dxx) think of re-using the previous output (e.g. dx)
    '''
    # YOUR CODE HERE
    dx = cv2.Sobel(bw_img, cv2.CV_64F, 1, 0, ksize = 3)
    dy = cv2.Sobel(bw_img, cv2.CV_64F, 0, 1, ksize= 3)

    dxx = np.square(dx)
    dyy = np.square(dy)
    dxy = dx*dy

    sum_mask = np.matrix([[1,1,1], [1,1,1], [1,1,1]])
    sumdxx = scipy.ndimage.convolve(dxx, sum_mask, mode='constant')
    sumdyy = scipy.ndimage.convolve(dyy, sum_mask, mode='constant')
    sumdxy = scipy.ndimage.convolve(dxy,sum_mask, mode='constant')

    det = sumdxx*sumdyy - np.square(sumdxy)
    trace = sumdxx + sumdyy
    R = det - k*np.square(trace)
    
    '''
    HINTS:
    1.> Think of R = det - trace * k
    2.> We can use ndimage.convolve
    3.> sort (argsort) the values and pick the alpha larges ones
    3.> points_of_interest should have this structure [[x1,x2,x3...],[y1,y2,y3...]] (2,alpha)
        where x_i is across H and y_i is across W
    '''
    # YOUR CODE HERE
    flattened_indices = np.argsort(R, axis = None)
    index_lists = np.unravel_index(flattened_indices, R.shape)
    x_list = index_lists[0][-alpha:]
    y_list = index_lists[1][-alpha:]
    points_of_interest = np.vstack([x_list, y_list])
    # print(points_of_interest.shape)
    
    
    ######### Actual Harris #########
    return points_of_interest

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
    i, alpha, image_path = args
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    f_name = 'tmp/%05d.npy' % i
    
    # ----- TODO -----
    '''
    HINTS:
    1.> Create a tmp dir to store intermediate results.
    2.> Read the image from image_path using skimage
    3.> extract filter responses and points of interest
    4.> store the response of filters at points of interest 
    '''
    image = io.imread(image_path)
    image = image.astype('float')/255

    filter_stack = extract_filter_responses(image)
    points_of_interest = get_harris_corners(image, alpha, 0.05)
    dictionary = np.array([])
    for j in range(alpha):
        fr_at_point = filter_stack[points_of_interest[0][j], points_of_interest[1][j], :]
        dictionary = np.vstack([dictionary, fr_at_point]) if dictionary.size else fr_at_point

    tgt_path = './tmp/' + str(i) + '.npy'
    np.save(tgt_path, dictionary)



    # YOUR CODE HERE

def compute_dictionary(num_workers=4):
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
    alpha = 150
    n_clusters = 200

    image_names = train_data['files']
    num_images = image_names.shape[0]

    for i in range(num_images):
        full_image_name = './data/' + image_names[i]
        list_of_args.append([i, alpha, full_image_name])
    
    
    with multiprocess.Pool(num_workers) as p:
        p.map(compute_dictionary_one_image, list_of_args)
    
    '''
    HINTS:
    
    1.> Use multiprocessing for parallel processing of elements
    2.> Next, load the tmp files and stack the responses stored as npy
    '''
    # YOUR CODE HERE
    filter_responses = np.array([])
    for i in range(num_images):
        arri = np.load('./tmp/' + str(i) +'.npy')
        filter_responses = np.vstack([filter_responses, arri]) if filter_responses.size else arri
    # raise NotImplementedError()
    # filter_responses = np.concatenate(filter_responses, axis=0)
    
    '''
    HINTS:
    1.> use sklearn.cluster.KMeans for clustersing
    2.> dictionary will be the cluster_centers_
    '''
    # YOUR CODE HERE
    KM = sklearn.cluster.KMeans(n_clusters).fit(filter_responses)
    dictionary = KM.cluster_centers_
    # raise NotImplementedError()
    np.save('dictionary.npy', dictionary)

    
# NOTE: comment out the lines below before submitting to gradescope

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    '''
    HINTS:
    (1) Use scipy.spatial.distance.cdist to find closest match from dictionary
    (2) Wordmap represents the indices of the closest element (np.argmin) in the dictionary
    '''
    filter_responses = extract_filter_responses(image)
    
    h, w, _ = filter_responses.shape
    filter_responses = np.reshape(filter_responses, [-1, filter_responses.shape[-1]])
    # ----- TODO -----
    
    
    # YOUR CODE HERE
    dist_matrix = scipy.spatial.distance.cdist(filter_responses, dictionary)
    sort_idx = np.argsort(dist_matrix, axis = 1)
    nearest_idx = sort_idx[:,0]
    wordmap = nearest_idx.reshape(h,w)

#     raise NotImplementedError()
    return wordmap 



if __name__=='__main__':
    # image_data = np.load('./data/train_data.npz')
    # image_names = image_data['files']
    # dictionary = np.load('dictionary.npy')
    # for t in range(10,15):
    #     image_path = './data/' + image_names[t]
    #     image = io.imread(image_path).astype('float')/255

    #     wordmap = get_visual_words(image, dictionary)
    #     save_wordmap(wordmap, 'wordmap' + str(t))

    # path_img = "./data/aquarium/sun_aydaknxraiwghvmi.jpg"
    # image = io.imread(path_img)
    # image = image.astype('float')/255

    # filter_responses = extract_filter_responses(image)
    # display_filter_responses(filter_responses)

    compute_dictionary(16)
