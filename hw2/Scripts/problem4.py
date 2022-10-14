import nbimporter
import numpy as np
import scipy.ndimage
from skimage import io
import skimage.transform
import os,time
import util
import multiprocess
import threading
import queue
import torch
import torchvision
import torchvision.transforms
from pytorch_pretrained_vit import ViT

def multichannel_conv2d(x,weight,bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H,W,input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H,W,output_dim)
    '''
    h, w, input_dims = x.shape
    output_dims = weight.shape[0]
    final_res = np.zeros((h, w, output_dims))
    '''
    HINTS:
    1.> for 2D convolution we need to use np.fliplr and np.flipud
    2.> can use scipy.ndimage.convolve with the flipped kernel
    3.> don't forget to add the bias
    '''
    # YOUR CODE HERE
    for j in range(output_dims):
        for k in range(input_dims):
            filt = weight[j,k,:,:]
            filt = np.flipud(filt)
            filt = np.fliplr(filt)
            final_res[:,:,j] = final_res[:,:,j] + scipy.ndimage.convolve(x[:,:,k], filt, mode='constant')
        final_res[:,:,j] = final_res[:,:,j] + bias[j]
#     raise NotImplementedError()
    return final_res


def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''
    # YOUR CODE HERE
    y = np.maximum(x, 0)
    return y

def max_pool2d(x,size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H,W,input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size,W/size,input_dim)
    '''
    h, w, dims = x.shape
    '''
    HINTS:
    1.> estimate the shape you need to apply the pooling operation.
    2.> We can smart fill the padding with np.nan and then use np.nanmax to select the max (avoiding nan)
    3.> We can input the grid (start_x:end_x, start_y:end_y, dim) as smart array indexing to np.nanmax
    '''
    # YOUR CODE HERE
    pooled_arr = np.zeros([int(h/size), int(w/size), dims])
    for i in range(int(h/size)):
        for j in range(int(w/size)):
            pooled_arr[i,j,:] = np.max(x[i*size:(i+1)*size, j*size:(j+1)*size], axis=(0,1))
    return pooled_arr

def linear(x,W,b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    
    # YOUR CODE HERE
    y = np.matmul(W, x) + b
    return y


def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H,W,3)

    [output]
    * image_processed: torch.array of shape (3,H,W)
    '''

    # ----- TODO -----
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(image.shape == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]
    '''
    HINTS:
    1.> Resize the image (look into skimage.transform.resize)
    2.> normalize the image
    3.> convert the image from numpy to torch
    '''
    # YOUR CODE HERE
    image = skimage.transform.resize(image, (224,224))
    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]
    image_processed = (image-mean)/std
    image_processed = torchvision.transforms.ToTensor()(image_processed)
    return image_processed


def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H,W,3)
    * vgg16_weights: list of shape (L,3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''
    
    feat = np.copy(x)
    # YOUR CODE HERE
    linear_count = 0
    for v in vgg16_weights:
        if v[0] == 'conv2d':
            weights = v[1]
            bias = v[2]
            feat = multichannel_conv2d(feat, weights, bias)
        if v[0] == 'relu':
            feat = relu(feat)
        if v[0] == 'maxpool2d':
            size = v[1]
            feat = max_pool2d(feat, size)
        if v[0] == 'linear':
            weights = v[1]
            bias = v[2]
            if linear_count ==0:
                feat = np.transpose(feat, (2,0,1)).flatten()
            feat = linear(feat, weights, bias)
            linear_count += 1
            if linear_count == 2:
                break
        
    # raise NotImplementedError()
    return feat

def evaluate_deep_extractor(img, vgg16):
    '''
    Evaluates the deep feature extractor for a single image.

    [input]
    * image: numpy.ndarray of shape (H,W,3)
    * vgg16: prebuilt VGG-16 network.

    [output]
    * diff: difference between the two feature extractor's result
    '''
    
    vgg16_weights = util.get_VGG16_weights()
    img_torch = preprocess_image(img)
    
    feat = extract_deep_feature(np.transpose(img_torch.numpy(), (1,2,0)), vgg16_weights)
    
    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
    
    return np.sum(np.abs(vgg_feat_feat.numpy() - feat))

# NOTE: comment out the lines below before submitting to gradescope
# Visible test cases (for debugging)
# path_img = "./data/aquarium/sun_aztvjgubyrgvirup.jpg"
# image = io.imread(path_img)
# image = image.astype('float') / 255

# vgg16 = torchvision.models.vgg16(pretrained=True).double()
# vgg16.eval()
# error = evaluate_deep_extractor(image, vgg16)

# # This error should be less than 1e-10
# print("Error:", error)

def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''
    i, image_path, vgg16 = args
    # print('Training Image - ' + str(i))
    image = io.imread(image_path).astype('float')/255
    
    '''
    HINTS:
    1.> Think along the lines of evaluate_deep_extractor
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    img_torch = preprocess_image(image)
    
    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
        
#     raise NotImplementedError()
    return [i,vgg_feat_feat.numpy()]

def build_recognition_system(vgg16, num_workers=16):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("./data/train_data.npz")
    '''
    HINTS:
    1.> Similar approach as Q1.2.2 and Q3.1.1 (create an argument list and use multiprocessing)
    2.> Keep track of the order in which input is given to multiprocessing
    '''
    # YOUR CODE HERE
    image_names = train_data['files']
    labels = train_data['labels']
    image_num = image_names.shape[0]
    
    lists_of_args = []
    
    for i in range(image_num):
        full_image_name = './data/' + image_names[i]
        lists_of_args.append([i, full_image_name, vgg16])
        
    with multiprocess.Pool(num_workers) as p:
        output = p.map(get_image_feature, lists_of_args)
    
    features = [x[1] for x in output]
    
#     raise NotImplementedError()
    ordered_features = [None] * len(features)
    '''
    HINTS:
    1.> reorder the features to their correct place as input
    '''
    # YOUR CODE HERE
    ordered_features = np.array(features)
#     raise NotImplementedError()
    print("done", ordered_features.shape)
    
    np.savez('trained_system_deep.npz', features=ordered_features, labels=labels)


def helper_func(args):
    # YOUR CODE HERE
    
    i, image_path, vgg16, trained_features, train_labels = args
    print('Test image - ' + str(i))
    feat = get_image_feature([i, image_path, vgg16])[1]
    print(feat.shape)
    err = np.square(np.abs(trained_features - feat))
    print(err.shape)
    sum_err = np.sum(err, axis = 1)
    print(sum_err.shape)
    idx = np.argmin(sum_err)
    pred_label = train_labels[idx]
    
    
#     raise NotImplementedError()
    return [i, pred_label]


def evaluate_recognition_system(vgg16, num_workers=16):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    '''
    HINTS:
    (1) Students can write helper functions (in this cell) to use multi-processing
    '''
    test_data = np.load("./data/test_data.npz")

    # ----- TODO -----
    trained_system = np.load("trained_system_deep.npz")
    image_names = test_data['files']
    test_labels = test_data['labels']

    trained_features = trained_system['features']
    train_labels = trained_system['labels']

    print("Trained features shape: ", trained_features.shape)
    
    '''
    HINTS:
    1.> [Important] Can write a helper function in this cell of jupyter notebook for multiprocessing
    
    2.> Helper function will compute the vgg features for test image (get_image_feature) and find closest
        matching feature from trained_features.
    
    3.> Since trained feature is of shape (N,K) -> smartly repeat the test image feature N times (bring it to
        same shape as (N,K)). Then we can simply compute distance in a vectorized way.
    
    4.> Distance here can be sum over (a-b)**2
    
    5.> np.argmin over distance can give the closest point
    '''
    # YOUR CODE HERE
    lists_of_args = []
    num_test = image_names.shape[0]
    for i in range(num_test):
        full_image_name = './data/' + image_names[i]
        lists_of_args.append([i, full_image_name, vgg16, trained_features, train_labels])

    with multiprocess.Pool(num_workers) as p:
        output = p.map(helper_func, lists_of_args)

    ordered_labels = np.array([x[1] for x in output])
#     raise NotImplementedError()

    print("Predicted labels shape: ", ordered_labels.shape)
    
    '''
    HINTS:
    1.> Compute the confusion matrix (8x8)
    '''
    # YOUR CODE HERE
    test_labels = np.array(test_labels, dtype = int)
    conf_matrix = np.zeros([8,8])
    for i in range(len(test_labels)):
        conf_matrix[test_labels[i], ordered_labels[i]] = conf_matrix[test_labels[i], ordered_labels[i]] + 1

    accuracy = np.trace(conf_matrix)/np.sum(conf_matrix)
#     raise NotImplementedError()
    
    np.save("./trained_conf_matrix.npy",conf_matrix)
    return conf_matrix, accuracy
    # pass

# NOTE: comment out the lines below before submitting to gradescope
### Run the code
# if __name__ == '__main__':
#     vgg16 = torchvision.models.vgg16(pretrained=True).double()
#     vgg16.eval()

#     print('Done')

#     # build_recognition_system(vgg16)
#     conf_matrix, accuracy = evaluate_recognition_system(vgg16)
#     # We expect the accuracy to be greater than 0.9
#     print("Accuracy:", accuracy)




# loads a pretrained ViT
vit_model = ViT('B_16_imagenet1k', pretrained=True)
vit_model.eval()

# iterate through the submodules and print out names
# for name, module in vit_model.named_modules():
#     print(name)

class FeatureExtractor:
    '''
    A class that takes in a nn.Module model and extracts feature from specified layer name.
    '''
    def __init__(self, model, layername='transformer'):
        self.extracted_feature = None
        self.model = model              # This will be vit_model in our case
        self.layername = layername

        # Apply hook to the transformer module
        '''
        HINTS:
        1.> The for loop of named_modules() we provided will be useful here.
        2.> Apply feature_extract_hook() to the transformer module using register_forward_hook()
        '''
        # YOUR CODE HERE
        for name, module in model.named_modules():
            if name == layername:
                mod_of_int = module
        h = model.transformer.register_forward_hook(self.feature_extract_hook(mod_of_int))

        # raise NotImplementedError()

    def feature_extract_hook(self, module):
        '''
        A function hook that extracts the module's output to the global variable `extracted_feature`

        [input]
        * module: module of interest
        * input: input of the module
        * output: output of the module
        '''

        '''
        HINTS:
        1.> You don't need to use all the arguments in this function.
        2.> What you need to do in this function should be really simple.
        '''
        # YOUR CODE HERE
        def fun(module, input, output):
            self.extracted_feature = output
        return fun
        # raise NotImplementedError()

    def extract_feature(self, img):
        '''
        Takes in an image, feed it to the model, and outputs the desired feature.
        
        [input]
        * x: preprocessed image
        
        [output]
        * feature: feature extracted from the specified layer name
        '''
        x = preprocess_image_vit(img).float()
        
        # simply run a forward pass of the model
        with torch.no_grad():
            self.model(x.unsqueeze(0))

        # feature will be extracted in self.extracted_feature already, thanks to the hook
        # you might wonder why we take only part of the output as feature
        # this is because we are only using the "class token" as the feature
        # for more details, please read the paper!
        return self.extracted_feature.numpy()[0, 0]


def preprocess_image_vit(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H,W,3)

    [output]
    * image_processed: torch.array of shape (3,H,W)
    '''

    # ----- TODO -----
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(image.shape == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]
    '''
    HINTS:
    1.> This function is essentially the same as the one you made before
    2.> Make sure you change the image size, mean, and std.
    '''
    # YOUR CODE HERE
    image = skimage.transform.resize(image, (384,384))
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    image_processed = (image-mean)/std
    image_processed = torchvision.transforms.ToTensor()(image_processed)
    return image_processed

def build_recognition_system_vit(vit_feat_extractor):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vit_feat_extractor: feature extractor for ViT

    [saved]
    * features: numpy.ndarray of shape (N,K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("./data/train_data.npz", allow_pickle=True)
    '''
    HINTS:
    1.> Similar approach as Q4.2.1
    2.> Do a for loop here instead of multiprocessing (it can take around 30 min to run)
    '''
    # YOUR CODE HERE
    image_names = train_data['files']
    labels = train_data['labels']
    image_num = image_names.shape[0]

    features = np.array([])

    for i in range(image_num):
        full_path = './data/' + image_names[i]
        img = io.imread(full_path).astype('float')/255
        feat = vit_feat_extractor.extract_feature(img)
        features = np.vstack([features, feat]) if features.size else feat

    labels = np.array(labels)
    
    np.savez('trained_system_vit.npz', features=features, labels=labels)

def evaluate_recognition_system_vit(vit_feat_extractor):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vit_feat_extractor: feature extractor for ViT

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("./data/test_data.npz", allow_pickle=True)

    # ----- TODO -----
    trained_system = np.load("trained_system_vit.npz", allow_pickle=True)
    image_names = test_data['files']
    test_labels = test_data['labels']

    trained_features = trained_system['features']
    train_labels = trained_system['labels']

    print("Trained features shape: ", trained_features.shape)
    
    '''
    HINTS:
    1.> Similar approach as Q4.2.1
    2.> Do a for loop here instead of multiprocessing
    '''
    # YOUR CODE HERE
    num_test = image_names.shape[0]
    ordered_labels = np.array([], dtype = int)

    for i in range(num_test):
        print('Test ' + str(i))
        img_path = './data/' + image_names[i]
        img = io.imread(img_path).astype('float')/255
        test_feat = vit_feat_extractor.extract_feature(img)
        nearest_idx = np.argmin(np.sum(np.square(trained_features - test_feat), axis=1))
        ordered_labels = np.append(ordered_labels, train_labels[nearest_idx])

    # raise NotImplementedError()

    print("Predicted labels shape: ", ordered_labels.shape)
    
    '''
    HINTS:
    1.> Same code as Q4.2.1, just copy it over
    '''
    # YOUR CODE HERE
    test_labels = np.array(test_labels, dtype = int)
    conf_matrix = np.zeros([8,8])
    for i in range(len(test_labels)):
        conf_matrix[test_labels[i], ordered_labels[i]] = conf_matrix[test_labels[i], ordered_labels[i]] + 1

    accuracy = np.trace(conf_matrix)/np.sum(conf_matrix)
    # raise NotImplementedError()
    
    np.save("./trained_conf_matrix_vit.npy",conf_matrix)
    return conf_matrix, accuracy
    # pass

vit = ViT('B_16_imagenet1k', pretrained=True)
vit.eval()
vit_feat_extractor = FeatureExtractor(vit)
# build_recognition_system_vit(vit_feat_extractor)

conf_matrix, accuracy = evaluate_recognition_system_vit(vit_feat_extractor)
print("Accuracy:", accuracy)