# DO NOT MODIFY! helper functions and constants
import nbimporter
from p1 import cv2, np, plt, math
from p1 import get_parameters, Gauss2D, filter_image_vec
from p2 import edge_detection_nms

image_list, constants = get_parameters()

#----------------------------------------------------------------------
def visualize(image_name, constants):
    image_rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    print("-" * 50 + "\n" + "Original Image:")
    plt.imshow(image_rgb); plt.show() # Displaying the sample image
    
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_m, image_o, image_x, image_y = edge_detection_nms(image, constants.sigma)
    
    print("-" * 50 + "\n" + "Edge Image:")
    plt.imshow(image_m, cmap="gray"); plt.show() # Displaying the sample image
    
    image_m_thres = 1.0*(image_m > constants.thres) # threshold the edge magnitude image
    print("-" * 50 + "\n" + "Thresholded Edge Image:")
    plt.imshow(image_m_thres, cmap="gray"); plt.show() # Displaying the sample image
    
    #--------------hough transform----------------
    H, rho_arr, theta_arr = hough_transform(image_m, constants.thres, constants.rho_res, constants.theta_res)   
    peak_rho_arr, peak_theta_arr = peak_hough_lines(H, rho_arr, theta_arr, constants.num_lines)
    
    #--------------vis using infinitely long lines----------------------------
    vis_line_len = 1000 # len of line in pixels, big enough to span the image
    vis_image_rgb = np.copy(image_rgb)
    for (rho, theta) in zip(peak_rho_arr, peak_theta_arr):
        x0 = rho*np.cos(theta); y0 = rho*np.sin(theta)
        x1 = int(x0 - vis_line_len*np.sin(theta)); y1 = int(y0 + vis_line_len*np.cos(theta))
        x2 = int(x0 + vis_line_len*np.sin(theta)); y2 = int(y0 - vis_line_len*np.cos(theta)); 
        cv2.line(vis_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    #---------------------------------------------
    print("-" * 50 + "\n" + "Edge Image:")
    plt.imshow(vis_image_rgb); plt.show() # Displaying the sample image
    
    return

def hough_transform(image_m, thres, rho_res, theta_res):
    """Compute Hough Transform for the image

    Args:
        image_m: np.array, HxW, edge magnitude image.
        thres: float, scalar to threshold image_m
        rho_res: integer, resolution of rho
        theta_res: integer, resolution of theta in degrees
        
    Returns:
        H: np.array, (num of rhos x num of thetas), hough transform accumulator (rho x theta), NOT theta x rho!
        rho_arr: np.array, dim=num of rhos, quantized rhos
        theta_arr: np.array, dim=num of thetas, quantized thetas
    """
    
    image_m_thres = 1.0*(image_m > thres) # threshold the edge magnitude image
    height, width = image_m_thres.shape # image height and width 
    diagonal_len = np.ceil(np.sqrt(height**2 + width**2)) # image diagonal = rho_max   
    # compute rho_arr, we go from [-rho_max to rho_max] in rho_res steps
    # rho_arr = ?
    # YOUR CODE HERE
    rho_arr = np.arange(-diagonal_len, diagonal_len + rho_res, rho_res)
    
    # compute theta_arr, we go from [0, pi] in theta_res steps, NOT [-pi/2, pi/2]!
    # Note theta_res is in degrees but theta_scale should be in radians [0, pi]
    # theta_arr = ?
    # YOUR CODE HERE
    theta_arr = np.deg2rad(np.arange(0,180 + theta_res,theta_res))

    
    H = np.zeros((len(rho_arr), len(theta_arr)), dtype=np.int32)
    
    # find all edge (nonzero) pixel indexes
    y_idxs, x_idxs = image_m_thres.nonzero() 
    
    for x, y in zip(x_idxs, y_idxs):
        for theta_idx, theta in enumerate(theta_arr):
            # compute rho_idx, note, theta is in radians!
            # Hint: compute rho first from theta, round it to nearest rho_prime in rho_arr
            # and then find rho_prime's rho_idx (index of rho_prime in rho_arr, NOT index of rho!)
            # rho_idx = ?
            # YOUR CODE HERE
            rho_act = x*np.cos(theta) + y*np.sin(theta)
            rho_idx = np.argmin(np.abs(rho_arr-rho_act))
            
            
            H[rho_idx, theta_idx] += 1

    
    return H, rho_arr, theta_arr

def peak_hough_lines(H, rho_arr, theta_arr, num_lines):
    """Returns the rhos and thetas corresponding to top local maximas in the accumulator H

    Args:
        H: np.array, (num of rhos x num of thetas), hough transform accumulator
        rho_arr: np.array, dim=num of rhos, quantized rhos
        theta_arr: np.array, dim=num of thetas, quantized thetas
        num_lines: integer, number of lines we wish to detect in the image
        
    Returns:
        peak_rho_arr: np.array, dim=num_lines, top num_lines rhos by votes in the H
        peak_theta_arr: np.array, dim=num_lines, top num_lines thetas by votes in the H
    """
    
    # compute peak_rho_arr and peak_theta_arr
    # sort H using np.argsort, pick the top num_lines lines
    # peak_rho_arr = ?
    # peak_theta_arr = ?
    # YOUR CODE HERE
    top_indices = np.argsort(H, None)
    top_rho_list, top_theta_list = np.unravel_index(top_indices, H.shape)
    top_rho_list = top_rho_list[-num_lines:]
    top_theta_list = top_theta_list[-num_lines:]
    peak_rho_arr = rho_arr[top_rho_list]
    peak_theta_arr = theta_arr[top_theta_list]

        
    
    assert(len(peak_rho_arr) == num_lines)
    assert(len(peak_theta_arr) == num_lines)
    return peak_rho_arr, peak_theta_arr

# image_idx = np.random.randint(0, len(image_list))
# visualize(image_list[image_idx], constants)