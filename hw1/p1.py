# DO NOT MODIFY! helper functions and constants
import cv2
import os
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib
from scipy import signal
from types import SimpleNamespace
import time

#----------------------------------------------------------------------
def Gauss2D(kernel=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in kernel]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

#----------------------------------------------------------------------
def visualize(function, image_name, kernel_size=(5, 5)):
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    print("-" * 50 + "\n" + "Original Image:")
    plt.imshow(image); plt.show() # Displaying the sample image
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Convert image to grayscale
    h_filter = Gauss2D(kernel_size, constants.sigma)
    
    image_filtered = function(image, h_filter) # testing
    print("-" * 50 + "\n" + "Filtered Image:")
    plt.imshow(image_filtered, cmap="gray"); plt.show()
    
    reference_image_filtered = signal.convolve2d(image, h_filter, mode="same")
    print("-" * 50 + "\n" + "Reference Filtered Image:")
    plt.imshow(reference_image_filtered, cmap="gray"); plt.show()
        
    return

#----------------------------------------------------------------------
def test(function, image_name, kernel_size=(5, 5), threshold=1e-12):
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY) # Convert image to grayscale
    h_filter = Gauss2D(kernel_size, constants.sigma)
    sobel_filter_x = np.matrix([[-1,0,1], [-2,0,2], [-1,0,1]])
    image_filtered = function(image, sobel_filter_x) # testing 
    reference_image_filtered = signal.convolve2d(image, sobel_filter_x, mode="same")

    error_arr = np.abs(reference_image_filtered - image_filtered)
    print('error: {}'.format(error_arr.mean()))
    assert(error_arr.mean() < threshold)
    return error_arr.mean()

#----------------------------------------------------------------------
def get_parameters():
    ##----------------------------------
    datadir     = 'data'    # the directory containing the images
    resultsdir  = 'results'  # the directory for dumping results

    ##-----------parameters------------
    constants = SimpleNamespace()
    constants.sigma      = 1.5
    constants.rho_res    = 1
    constants.theta_res  = 1
    constants.thres      = 30
    constants.num_lines  = 50

    image_list = []
    for filename in glob.glob(datadir+'/*.jpg'):
        image_list.append(filename)

    image_list.sort()
    return image_list, constants

#----------------------------------------------------------------------
image_list, constants = get_parameters()

def filter_image(image, h_filter):
    """Conduct convolutional filtering on the input image. 
    This function assumes that the input filter size is an odd number, 
    Pad the input image with 'constant' padding. 

    Args:
      image: np.array, HxW, the input grayscale image. 
      h_filter: np.array, KxK where K is the kernel size, the input image filter, created by the Gauss2D function. 

    Returns:
      image_output: np.array, HxW, the filtered image
    """
    start_time = time.time()
    row, col = image.shape
    h_row, h_col = h_filter.shape
    
    assert(h_row % 2 == 1 and h_col % 2 == 1)
    image = image.astype(np.float64) # uint8 -> float64
    image_output = np.empty_like(image) # output

    # Computes the row and col of padding needed
    row_padding = int(np.floor(h_row/2))
    col_padding = int(np.floor(h_col/2))
    
    # Pad the input image with zeros using the np.pad function. 
    # After this step, you should be able to perform convolution with the`same` padding scheme
    # Refer: https://www.geeksforgeeks.org/types-of-padding-in-convolution-layer/
    # image_pad = np.pad(?)
    # YOUR CODE HERE
    #print(image[0:10,0:10])
    image = np.pad(image, [(row_padding, row_padding), (col_padding, col_padding)], mode='constant')

    #raise NotImplementedError()
    
    h_filter_flipped = np.flip(h_filter)
    # Perform convolution on the padded image. 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # image_output[i, j] = ?
            # YOUR CODE HERE
            if i < row_padding or j < col_padding:
                continue
            elif i >= row+row_padding or j >= col+col_padding:
                continue
            else:
                image_patch = image[i-row_padding:i+row_padding+1, j-col_padding:j+col_padding+1]
                val = np.sum(np.multiply(image_patch, h_filter_flipped))
                image_output[i-row_padding,j-col_padding] = val

            #raise NotImplementedError()

    return image_output

def filter_image_vec(image, h_filter):
    """Conduct convolutional filtering using vectorization on the input image

    Args:
      image: np.array, HxW, the input grayscale image. 
      h_filter: np.array, KxK where K is the kernel size, the input image filter, created by the Gauss2D function.

    Returns:
      image_output: np.array, HxW, the filtered image
    """
    row, col = image.shape
    h_row, h_col = h_filter.shape
    
    assert(h_row % 2 == 1 and h_col % 2 == 1)
    image = image.astype(np.float64) # uint8 -> float64

    # Computes the row and col of padding needed
    row_padding = int(np.floor(h_row/2))
    col_padding = int(np.floor(h_col/2))
    
    # Pad the input image with zeros same as before!
    # image_pad = np.pad(?)
    # YOUR CODE HERE
    image = np.pad(image, [(row_padding, row_padding), (col_padding, col_padding)], mode='constant')
   
    #raise NotImplementedError()
    h_filter_flipped = np.flip(h_filter)
    h_flipped_opened = h_filter_flipped.reshape(-1,1)
    windows = []
    # Perform convolution on the padded image. 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # what is our convolution window in the image?
            # window = ?
            if i <row_padding or j < col_padding:
                continue
            elif i >= row+row_padding or j>= col + col_padding:
                continue
            else:
                window = image[i-row_padding:i+row_padding+1, j-col_padding:j+col_padding+1]
                windows.append(window.reshape(-1, 1))
            # YOUR CODE HERE
            #raise NotImplementedError()
            
    
    image_to_convolve = np.concatenate(windows, axis=1)  ## (h_row*h_col) x num_pixels
    
    # Perform matrix multiplication between the image_to_convolve and h_filter followed by reshape!
    # image_filter = ?
    # YOUR CODE HERE
    output = np.matmul(np.transpose(h_flipped_opened), image_to_convolve)
    image_output = np.reshape(output, (row, col))
    #raise NotImplementedError()
    return image_output

# image_idx = np.random.randint(0, len(image_list))
# visualize(filter_image, image_list[image_idx])
# error = test(filter_image, image_list[image_idx])
# print('image_idx:{} error:{}'.format(image_idx, error))