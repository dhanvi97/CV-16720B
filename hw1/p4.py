## DO NOT MODIFY! 
## Import from previous notebook
import nbimporter
from p1 import cv2, np, plt, math, SimpleNamespace
from p1 import get_parameters, Gauss2D, filter_image_vec
from p2 import edge_detection_nms
from p3 import hough_transform, peak_hough_lines

image_list, constants = get_parameters()

#----------------------------------------------------------------------
# Different from visualize in p3, calls hough_accumulator_nms()
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
    H = hough_accumulator_nms(H) # nms on H
    peak_rho_arr, peak_theta_arr = peak_hough_lines(H, rho_arr, theta_arr, constants.num_lines)
    
    #--------------vis----------------------------
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

def hough_accumulator_nms(H):
    """Compute Hough Transform for the image

    Args:
        image_m: np.array, HxW, edge magnitude image.
        
    Returns:
        image_m_prime: np.array, HxW, suppressed edge magnitude image.
    """
    H_prime = np.copy(H) 
    H_pad = np.pad(H, 1)
    neighbor_offsets = [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2) if (dy != 0 or dx != 0)]
    
    # compute supression mask per neighbour, 1 to suppress, 0 to keep
    # compare H and a part of H_pad, the part of H_pad can be obtained by moving H_pad using the neighbor_offsets
    # suppress_masks_per_neighbor = [? for (dy, dx) in neighbor_offsets]
    # YOUR CODE HERE
    # H_row, H_col = H.shape
    suppress_masks_per_neighbor = [np.greater(H_pad[1 + dy: H.shape[0] + dy + 1, 1 + dx:H.shape[1] + dx + 1], H) for (dy, dx) in neighbor_offsets]
    suppress_mask = np.logical_or.reduce(suppress_masks_per_neighbor)
    H_prime[suppress_mask] = 0
    return H_prime

def visualize_line_segments(image_name, constants):
    image_rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)  
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_m, image_o, image_x, image_y = edge_detection_nms(image, constants.sigma)
    vis_image_rgb = np.copy(image_rgb)
    
    #--------------hough transform----------------
    H, rho_arr, theta_arr = hough_transform(image_m, constants.thres, constants.rho_res, constants.theta_res)
    H = hough_accumulator_nms(H) # nms on H
    peak_rho_arr, peak_theta_arr = peak_hough_lines(H, rho_arr, theta_arr, constants.num_lines)
    vis_line_len = 10
    rounding_thresh = 1
    
    # visualize line segments (not infinite lines!)
    # vis_image_rgb = ?
    # For each pixel in image - check if in edge using H
    # If yes - then fit small line segment centered at that pixel along line
    vis_line_len = 1000 # len of line in pixels, big enough to span the image
    blank_image = np.zeros(image_rgb.shape, dtype=np.uint8)

    image_m_thres = 1.0*(image_m > constants.thres)
    for (rho, theta) in zip(peak_rho_arr, peak_theta_arr):
        x0 = rho*np.cos(theta); y0 = rho*np.sin(theta)
        x1 = int(x0 - vis_line_len*np.sin(theta)); y1 = int(y0 + vis_line_len*np.cos(theta))
        x2 = int(x0 + vis_line_len*np.sin(theta)); y2 = int(y0 - vis_line_len*np.cos(theta)); 
        cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lines_image = cv2.cvtColor(blank_image, cv2.COLOR_RGB2GRAY)
    # plt.imshow(lines_image, cmap='gray'); plt.show()
    # print(lines_image.shape)

    lines_image_thresh = 1.0*(lines_image > 100)
    mask = np.logical_and(image_m_thres, lines_image_thresh)
    image_m_thres = np.multiply(image_m_thres, mask)
    invert_thres = 1-image_m_thres
    elem = np.argwhere(invert_thres == 0)
    for idx in elem:
        invert_thres[idx[0]-1:idx[0]+2, idx[1]-1:idx[1] + 2] = 0 

    vis_image_rgb[:,:,0] = np.multiply(vis_image_rgb[:,:,0], invert_thres)
    vis_image_rgb[:,:,2] = np.multiply(vis_image_rgb[:,:,2], invert_thres)

    vis_image_bgr = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)


    # print("-" * 50 + "\n" + "Edge Image:")
    # plt.imshow(vis_image_rgb); plt.show() # Displaying the sample image

    
    
    # YOUR CODE HEREt
    # raise NotImplementedError()
    
    return image_rgb, vis_image_bgr

## TOY TEST!
# H = np.random.rand(5, 5)
# H_prime = hough_accumulator_nms(H)
# print(H); print(H_prime)

# Uncomment to visualize
# image_idx = np.random.randint(0, len(image_list))
# visualize_line_segments(image_list[image_idx], constants)
for image_idx in range(len(image_list)):
    image_rgb, vis_image_bgr = visualize_line_segments(image_list[image_idx], constants)
    edge_image = r'EdgeImg' + str(image_idx) + '.png'
    cv2.imwrite(edge_image, vis_image_bgr)