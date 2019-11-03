#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a library gathering all (relevant) image processing functions developed 
within the scope of "Optimisation and Image Processing - OIP" 2018, MCI, SDU.

Put this library in your script folder and load by (for instance):
    
    import OIPlib_ImageProcessing as ip
    
apply functionality by (for instance)

    img, imgRED, imgGREEN, imgBLUE = ip.load_image_GUI()

Created on Tue Sep 11 11:17:33 2018

UPDATE_1, 2018-09-17: 
    ADDED LECTURE 3 CONTENT (LINEAR FILTERING) 
    
UPDATE_2, 2018-09-25:
    ADDED LECTURE 4 CONTENT / FUNCTIONALITY: 
        * NONLINEAR FILTERING     
        * Conversion imgBIN <-> PointSet
        * Basic Operations on PointSets (Union, Intersection, Translation, Reflection)
        
UPDATE_3, 2018-10-01:
    ADDED FUNCTIONS FOR BASIC MORPHOLOGICAL OPERATIONS ON IMAGES AND SETS. 
    
UPDATE_4, 2018-10-10:
    ADDED FUNCTIONS FOR:
        * Edge detection
        * Laplace Sharpening
        * Unsharp Masking
        
        * Thinning (SLOW - faster version yet to come...)
        
        * Hough Line Detection (Original + Accelerated / Vectorised)
       

@author: jost
"""


# ------------------------------------
# IMPORTS: 
# ------------------------------------

# needed almost every time: 
import matplotlib.pyplot as plt # for plotting
import matplotlib.image as mpimg # for image handling and plotting
import numpy as np # for all kinds of (accelerated) matrix / numerical operations
from scipy.signal import convolve2d

# tkinter interface module for GUI dialogues (so far only for file opening):
import tkinter as tk
from tkinter.filedialog import askopenfilename


def loadImage(imgUrl):
    """Load an image from a local directory.

    Args:
    
        imgUrl (string): The path to an image file.

    Returns:

        uint8Img (n-chan uint8 numpy array): A image with all channels.
    """
    floatImage = mpimg.imread(imgUrl)
    fileExtension = imgUrl.split(".")[-1].lower()
    # TIFF images define colors as an integer between 0 and 65535.
    if fileExtension == "tiff" or fileExtension == "tif":
        return (floatImage / 65535 * 255).astype(np.uint8)
    # PNG images define colors as a floating point number between 0 and 1.
    if fileExtension == "png":
        return (floatImage * 255).astype(np.uint8)
    return (floatImage).astype(np.uint8)

def showImage(uint8Img, title='Image', cmap='gray', vmin=0, vmax=255, figsize=5):
    """Display an image as a simple intensity map with a colorbar.

    Args:

        uint8Img (n-chan uint8 numpy array): An image to be displayed. May have more than a single channel. Be aware of the color map, when displaying multi-channel images.
        title (string): A title for the image.
        cmap (string): A colormap that defines the available color space.
        vmin (number): The lowest pixel value in the image.
        vmax (number): The highest pixel value in the image.
        figsize (number): The figure size in cm.
    
    Returns:

        fig (matplotlib figure): The figure object of the resulting plot.
        ax (matplotlib axes): The axes object of the resulting plot.
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    plot = ax.imshow(uint8Img, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(title)
    fig.colorbar(plot, ax=ax)    
    return fig, ax

def gray2Binary(uint8Img, threshold=128):
    """Creates a binary image by settings all pixels below the threshold to 0 and all other to 1.

    Args:

        uint8Img (1-chan uint8 numpy array): An grayscale image to be converted. May have only a single channel.
        threshold (number): Threshold value below which a pixel will be set to 0.

    Returns:

        uint8Img (1-chan uint8 numpy array): An single channel binary image, where each pixel has either the value 0 or 1.
    """
    return (uint8Img >= threshold).astype(np.uint8)

def binary2Set(uint8Img):
    """Converts a binary image to a set.

    Args:
        
        uint8Img (1-chan uint8 numpy array): A binary image, which contains only values of 0 or 1.

    Returns:

        setImg (number tuple set): A set of the foreground pixel coordinate tuples with a value of 1 in the binary image.
    """
    return { (x, y) for x, y in np.argwhere(uint8Img == 1) }

def set2Binary(imgSet, size=(9, 9)):
    """Converts a set to a binary image.

    Args:
        
        setImg (number tuple set): A set of the foreground pixel coordinate tuples with a value of 1 in the binary image.
        size (number tuple): The width w and height h of the desired image.

    Returns:

        uint8Img (1-chan uint8 numpy array): A binary image, which contains only values of 0 or 1.
    """
    uint8Img = np.zeros(size, np.uint8)
    for x, y in imgSet:
        if x < size[0] and y < size[1]:
            uint8Img[x, y] = 1
    return uint8Img

def createStructuringElement(radius=1, neighborhood="8N"):
    """Create a structuring element function based on the neighborhood and the radius.

    Args:

        radius (number): The radius of the structuring element excluding the center pixel.
        neighborhood (string): 4N or 8N neighborhood definition around the center pixel. 
    
    Returns:

        getStructuringElement (function): A function, which returns the neighborhood for a given center based on the configured radius and neighboorhood definition.
    """
    def getStructuringElement(center):
        """Create a set of pixel coordinates for all neighbor elements.
        
        Args:

            center (number tuple): A pixel coordinate tuple of the center pixel of the structuring element.

        Returns:

            setImg (number tuple set): A set of the foreground pixel coordinate tuples that make up the neighboorhood for the given center.
        """
        neighbors = set()
        if neighborhood == "4N":
            for x in range(center[0]-radius, center[0]+radius+1):
                for y in range(center[1]-radius, center[1]+radius+1):
                    if abs(center[0] - x) + abs(center[1] - y) <= radius:
                        neighbors.add((x, y))
        else:
            for x in range(center[0]-radius, center[0]+radius+1):
                for y in range(center[1]-radius, center[1]+radius+1):
                    neighbors.add((x, y))
        return neighbors
    # Use partial application of function arguments to dynamically calculate the neighborhood based on previous constraints.
    return getStructuringElement

def dilateSet(setImg, getStructuringElement=createStructuringElement()):
    """Dilates a set by a given structural element.

    Args:

        setImg (number tuple set): A set of the foreground pixel coordinate tuples with a value of 1 in the binary image.
        getStructuringElement (function): A function that returns the structuring element for a given center pixel.
    
    Returns:

        setImg (number tuple set): A set of the foreground pixel coordinate tuples with a value of 1 in the binary image.
    """
    # Rather slow (~4s). Parallelization via threads could help.
    dilatedSet = set()
    for x, y in setImg:
        dilatedSet.update(getStructuringElement((x, y)))
    return dilatedSet

def separateLegend(uint8Img, threshold = 127):
    """This separates the legend from the cluster image. It assumes the legend is horizontal. Thresholds the image to create a binary image, where the legend border is determined.
    
    Args: 
        
        img (numpy array): A grayscale image.
        threshold (number): The threshold factor in the binary conversion process.
    
    Returns:
        
        img (1-chan uint8 numpy array): The image without the legend.
        legend (1-chan uint8 numpy array): The legend of the image.
    """
    binImg = gray2Binary(uint8Img, threshold)
    condition = np.ones(binImg.shape[1], np.uint8)
    for i in range(binImg.shape[0]):
        if np.array_equal(binImg[i], condition):
            return uint8Img[:i,:], uint8Img[i:,:]

#------------------------------------------------------

def load_image_GUI():
    ''' This function loads an image, without a given path to the file, but by
    opening a GUI (based on the tkinter cross-platform interface). Should work on
    most OSs'''
    
    # GUI-based "getting the URL": 
    root = tk.Tk()
    root.filename = askopenfilename(initialdir = "../images",title = "choose your file",filetypes = (("png files","*.png"),("all files","*.*")))
    print ("... opening " + root.filename)
    imgURL = root.filename
    root.withdraw()
    
    # actually loading the image based on the aquired URL: 
    img=mpimg.imread(imgURL)
    img = img * 255
    img = img.astype(np.uint8)
    imgRED = img[:,:,0]
    imgGREEN = img[:,:,1]
    imgBLUE = img[:,:,2]
    return img, imgRED, imgGREEN, imgBLUE

def crop_levels(imgINT):
    '''helper function to crop all levels back into the 0...255 region'''
    imgINT[imgINT>=255]=255
    imgINT[imgINT<=0]=0
    return imgINT 

''' Convert RGB images to single channel grayscale images. '''

def rgb2GrayAverage(uint8Img):
    return ((uint8Img[:,:,0].astype(np.float)+uint8Img[:,:,1].astype(np.float)+uint8Img[:,:,1].astype(np.float))/3.0).astype(np.uint8)

def rgb2GrayLuminosity(uint8Img):
    """Convert RGB image to grayscale by using the luminosity method.

    Args:

        uint8Img (3-dim uint8 numpy array): A multi-channel RGB image.

    Returns:

        uint8Img (2-dim uint8 numpy array): A single-channel grayscale image.
    """
    return (0.21*uint8Img[:,:,0]+0.72*uint8Img[:,:,1]+0.07*uint8Img[:,:,2]).astype(np.uint8)

''' Convert grayscale images to binary images. '''





# ------------------------------------
# HISTOGRAM GENERATION: 
# ------------------------------------

def hist256(imgint8):
    ''' manual histogram creation for uint8 coded intensity images'''
    hist = np.zeros(255)
    for cnt in range(255):
         hist[cnt] = np.sum(imgint8==cnt)
    return(hist)
    
def cum_hist256(imgint8):
    ''' manual cumulative histogram creation for uint8 coded intensity images'''
    chist = np.zeros(255)
    for cnt in range(255):
         chist[cnt] = np.sum(imgint8<=cnt)
    return(chist)
    
# ----------------------------------------------------------------------------
# Point Operations: 
# ----------------------------------------------------------------------------
    
def threshold(imgINT,ath):
    imgTH = np.zeros(imgINT.shape)
    imgTH[imgINT >= ath] = 255
    return imgTH.astype(np.uint8)

def threshold2(imgINT,ath):
    return ((imgINT >= ath)*255).astype(np.uint8)
    

def threshold_binary(imgINT,ath):
    mask = imgINT >= ath
    imgTH = np.zeros(imgINT.shape)
    imgTH[mask] = 1
    return imgTH.astype(np.uint8)

def threshold_binary2(imgINT,ath):
    return imgINT >= ath

def adjust_brightness(img,a):
    imgB = crop_levels(img.astype(np.float)+a)
    return imgB.astype(np.uint8)

def adjust_contrast(img,a):
    imgC = crop_levels(img.astype(np.float)*a)
    return imgC.astype(np.uint8)

def invert_intensity(img):
    return 255-img

def auto_contrast256(img):
    alow = np.min(img)
    ahigh = np.max(img)
    amin = 0.
    amax = 255.
    return (amin + (img.astype(np.float) - alow) * (amax - amin) / (ahigh - alow)).astype(np.uint8)

def equalise_histogram256(img):
    M, N = img.shape
    H = cum_hist256(img)
    Hmat = np.zeros(img.shape)
    for cnt in range(255):
        Hmat[img==cnt] = H[cnt]
    imgEqHist = (Hmat * 255/(M*N)).astype(np.uint8)
    return imgEqHist

def shift_intensities(imgint8, source_int, target_int):
    img_out = imgint8
    img_out[img_out == source_int] = target_int  
    return img_out
    
# ----------------------------------------------------------------------------
# Create Noise: 
# ----------------------------------------------------------------------------

def add_salt_and_pepper(gb, prob):
    '''Adds "Salt & Pepper" noise to an image.
    gb: should be one-channel image with pixels in [0, 1] range
    prob: probability (threshold) that controls level of noise'''

    rnd = np.random.rand(gb.shape[0], gb.shape[1])
    noisy = gb.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy.astype(np.uint8)


# ----------------------------------------------------------------------------
# FILTER MATRICES:    
# ----------------------------------------------------------------------------
 
# Define some filter matrices: 
# The 3x3 averaging box:
Av3 = np.array([[1, 1, 1], 
                [1, 1, 1], 
                [1, 1, 1]])/9

# 5x5 Gaussian filter: 
Gauss5 = np.array([[0, 1, 2, 1, 0], 
                   [1, 3, 5, 3, 1], 
                   [2, 5, 9, 5, 2], 
                   [1, 3, 5, 3, 1], 
                   [0, 1, 2, 1, 0]])
Gauss5Norm = Gauss5/np.sum(Gauss5)

# 5x5 Mexican Hat Filter:
Mex5 = np.array([[0, 0, -1, 0, 0], 
                 [0, -1, 2, -1, 0], 
                 [-1, -2, 16, -2, -1], 
                 [0, -1, -2, -1, 0],
                 [0, 0, -1, 0, 0]])
Mex5Norm = Mex5/np.sum(Mex5)

# ----------------------------------------------------------------------------
# Edge Detection Filters: 

# Derivative:
HDx = np.array([[-0.5, 0., 0.5]])
HDy = np.transpose(HDx)

# Prewitt:
HPx = np.array([[-1., 0., 1.],
                [-1., 0., 1.],
                [-1., 0., 1.]])/6.
HPy = np.transpose(HPx)

# Sobel:
HSx = np.array([[-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]])/8.
HSy = np.transpose(HPx)


# Improved Sobel:
HISx = np.array([[-3., 0., 3.],
                [-10., 0., 10.],
                [-3., 0., 3.]])/32.
HISy = np.transpose(HPx)

# ----------------------------------------------------------------------------
# Laplace Filters: 

# Laplace:
HL4 = np.array([[0., 1., 0.],
               [1., -4., 1.],
               [0., 1., 0.]])
    
# Laplace 8:
HL8 = np.array([[1., 1., 1.],
                [1., -8., 1.],
                [1., 1., 1.]])
    
# Laplace 12:
HL12 = np.array([[1., 2., 1.],
                 [2., -12., 2.],
                 [1., 2., 1.]])

# ------------------------------------
# LINEAR FILTERING: 
# ------------------------------------

# ----------------------------------------------------------------------------
# "Naive" / pedestrian Filter implementation (we will rather use the convolutional approach)
def pedestrian_filter(imgINT, H):
    HX, HY = H.shape
    radiusX = np.floor_divide(HX,2)
    radiusY = np.floor_divide(HY,2)
    padded = np.pad(imgINT, ((radiusX, radiusX), (radiusY, radiusY)), 'reflect').astype(np.float)
    N, M = imgINT.shape
    filtered = imgINT.astype(np.float)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN+radiusX
            pidM = cntM+radiusY
            TMP = padded[pidN-radiusX:pidN+radiusX+1, pidM-radiusY:pidM+radiusY+1]
            filtered[cntN,cntM] = np.sum(np.multiply(TMP,H))
            
    return (crop_levels(filtered)).astype(np.uint8)


# ----------------------------------------------------------------------------
# Linear filtering by convolution
def conv2(x, y, mode='same'):
    # mimic matlab's conv2 function: 
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def filter_image(I,H):   
    # Convolution-based filtering: 
    Filtered = conv2(np.double(I),np.double(H));    
    # Reducing to original size and converting back to uint8: 
    # and CUT to the range between 0 and 255.
    return (crop_levels(Filtered)).astype(np.uint8)

def filter_image_float(I,H):   
    # Convolution-based filtering: 
    return conv2(np.double(I),np.double(H))   
    
# ----------------------------------------------------------------------------
# Gaussian Filter Matrix of arbitrary size: 
def create_gaussian_filter(fsize,sigma):
    '''
    Create a Gaussian Filter (square) matrix of arbitrary size. fsize needs 
    to be an ODD NUMBER!
    '''
    # find the center point:
    center = np.ceil(fsize/2)
    # create a "distance" vector (1xfsize matrix) from the center point: 
    tmp = np.arange(1,center,1)
    tmp = np.concatenate([tmp[::-1], [0], tmp])
    dist = np.zeros((1,tmp.shape[0]))
    dist[0,:] = tmp
    # create two 1D (x- and y-) Gaussian distributions: 
    Hgx = np.exp(-dist**2/(2*sigma**2))
    Hgy = np.transpose(Hgx)
    # build the outer product to get the full filter matrix: 
    HG = np.outer(Hgy, Hgx)
    # ... normalise... 
    SUM = np.sum(HG)
    HG = HG/SUM
    Hgx = Hgx/np.sqrt(SUM)
    Hgy = Hgy/np.sqrt(SUM)
    return HG, Hgx, Hgy, SUM

# ----------------------------------------------------------------------------
# Laplacian of Gaussian Filter matrix of arbitrary size: 
def create_LoG_filter(fsize,sigma):
    '''
    Create a Gaussian Filter (square) matrix of arbitrary size. fsize needs 
    to be an ODD NUMBER!
    '''
    # find the center point:
    center = np.floor(fsize/2)
    LoG = np.zeros((fsize, fsize))
    
    # create relative coordinates: 
    for cntN in range(fsize):
        for cntM in range(fsize):
            rad2 = (cntN-center)**2 + (cntM-center)**2
            LoG[cntN, cntM] = -1./(sigma**4) * (rad2-2.*sigma**2)*np.exp(-rad2/(2.*sigma**2))
    # SUM = np.sum(LoG)
    return LoG

# ------------------------------------
# NONLINEAR Filtering: 
# ------------------------------------

def max_filter(imgINT, radius):
    '''filter with the maximum (non-linear) filter of given radius'''
    padded = np.pad(imgINT, ((radius, radius), (radius, radius)), 'reflect').astype(np.float)
    N, M = imgINT.shape
    filtered = np.zeros(imgINT.shape)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN+radius
            pidM = cntM+radius
            filtered[cntN,cntM] = np.amax(padded[pidN-radius:pidN+radius+1, pidM-radius:pidM+radius+1])
    return filtered.astype(np.uint8)

def min_filter(imgINT, radius):
    '''filter with the minimum (non-linear) filter of given radius'''
    padded = np.pad(imgINT, ((radius, radius), (radius, radius)), 'reflect')#.astype(np.float)
    N, M = imgINT.shape
    filtered = np.zeros(imgINT.shape)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN+radius
            pidM = cntM+radius
            filtered[cntN,cntM] = np.floor(np.amin(padded[pidN-radius:pidN+radius+1, pidM-radius:pidM+radius+1]))
    return filtered.astype(np.uint8)

def calc_median(Hmat):
    '''calculate the median of an array...
    This is rather for "educational purposes", since numpy.median is available 
    and potentially (about 10x) faster. '''
    # Hvecsort = sorted([x for x in H.flat]) # LIST SORTING... (EVEN SLOWER!!!)
    Hvecsort = np.sort(Hmat.flat)          # NUMPY VECTOR SORTING...
    length = len(Hvecsort)  
    if (length % 2 == 0):      
        # if length is even, mitigate: 
        median = (Hvecsort[(length)//2] + Hvecsort[(length)//2-1]) / 2
    else:
        # otherwise just pick the center element: 
        median = Hvecsort[(length-1)//2]
    return median

def median_filter(imgINT, radius):
    padded = np.pad(imgINT, ((radius, radius), (radius, radius)), 'reflect').astype(np.float)
    N, M = imgINT.shape
    filtered = np.zeros(imgINT.shape)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN+radius
            pidM = cntM+radius
            filtered[cntN,cntM] = np.floor(np.median(padded[pidN-radius:pidN+radius+1, pidM-radius:pidM+radius+1]))
    return filtered.astype(np.uint8)

# ------------------------------------
# ------------------------------------
# Morphological Filtering: 
# ------------------------------------
# ------------------------------------

# ------------------------------------
# CONVERSION BETWEEN BINARY IMAGES AND POINT SETS: 
    
def PointSet_to_imgBIN(coordinates, imgShape):
    '''
    converting point sets to binary images of shape imgShape:
        INPUTS: 
            coordinates:    List of (2D) coordinates, e.g. [[0,0],[1,0],[2,2]]
            imgShape:       image shape, e.g. <matrix>.shape()
        OUTPUT: 
            image:          boolean matrix (with ones at the given coordinates), 
                            wish given shape imgShape
    '''  
    cmat = np.matrix(coordinates)
    image = (np.zeros(imgShape)).astype(np.bool)
 
    for i in range(len(coordinates)):
        if (cmat[i,0] in range(imgShape[0])) and (cmat[i,1] in range(imgShape[1])):
            image[cmat[i,0], cmat[i,1]] = 1 
        
    return image
    
def imgBIN_to_PointSet(imgBIN):
    '''
    converting binary images to point sets:
        INPUTS: 
            imgBIN:         boolean matrix 
        OUTPUT: 
            List of (2D) coordinates, e.g. [[0,0],[1,0],[2,2]]
    '''
    return (np.argwhere(imgBIN)).tolist()

# ----------------------------------
# Point Set Operations: 

def PL_union(listone, listtwo):
    ''' calculate the union of two point lists'''
    # convert to set of tuples: 
    l1_set = set(tuple(x) for x in listone)
    l2_set = set(tuple(x) for x in listtwo)
    # return the union as list of lists.
    return [ list(x) for x in l1_set | l2_set ]
    
def PL_intersect(listone, listtwo):
    ''' calculate the intersection of two point lists'''
    # convert to tuples:
    l1_set = set(tuple(x) for x in listone)
    l2_set = set(tuple(x) for x in listtwo)
    # return the intersection as list of lists: 
    return [ list(x) for x in l1_set & l2_set ]

def PL_translate(PointList,Point):
    ''' shift / translate a point list by a point'''
    return [([sum(x) for x in zip(Point, y)]) for y in PointList]

def PL_mirror(PointList):
    ''' mirror / reflect a point list'''
    return [list(x) for x in list((-1)*np.array(PointList))]

# ------------------------------------------------------
# MORPHOLOGICAL OPERATIONS ON BINARY IMAGES
# ------------------------------------------------------

def PL_dilate ( PointList_I , PointList_H ) : 
    DILATED_PointList = []
    for q in PointList_H:
        DILATED_PointList = PL_union(DILATED_PointList, PL_translate(PointList_I,q))
        
    return DILATED_PointList


def img_dilate(I,PLH):
    PLI = imgBIN_to_PointSet(I)
    return PointSet_to_imgBIN(PL_dilate(PLI, PLH), I.shape)


def img_erode(I,PLH):
    PLHFlip = PL_mirror(PLH)
    PLIinv = imgBIN_to_PointSet(1-I)
    PLI_Erode = PL_dilate(PLIinv,PLHFlip) 
    return 1-PointSet_to_imgBIN(PLI_Erode, I.shape)

def PL_erode(PointList_I , PointList_H): 
    # Up  for you to try :-) 
    return 

def img_BoundaryExtract(I , PLH) : # PUT YOUR CODE HERE!
    Ierode_inv = 1-img_erode(I, PLH)
    return PointSet_to_imgBIN(PL_intersect(imgBIN_to_PointSet(I),imgBIN_to_PointSet(Ierode_inv)), I.shape)

def img_open(I, PLH):
# PUT YOUR CODE HERE!
    return img_dilate(img_erode(I, PLH), PLH)

def img_close(I, PLH):
# PUT YOUR CODE HERE!    
    return img_erode(img_dilate(I, PLH), PLH)
# ------------------------------------
# COMMON STRUCTURING ELEMENTS (POINT SETS)

N4 = [[0,0],[-1,0],[1,0],[0,-1],[0,1]]

N8 = [[0,0],[-1,0],[1,0],[0,-1],[0,1],[-1,1],[1,-1],[1,1],[-1,-1]]

SmallDisk = [[0,0],[-1,0],[1,0],[0,-1],[0,1],[-1,1],[1,-1],[1,1],[-1,-1],
             [-2,-1],[-2,0],[-2,1],
             [2,-1],[2,0],[2,1],
             [-1,-2],[0,-2],[1,-2],
             [-1,2],[0,2],[1,2]]

# ----------------------------------------------------------------------------
# Edge Detection and Sharpening
# ----------------------------------------------------------------------------

def detect_edges(imgINT, Filter='Sobel'):
    Hx, Hy = {
        'Gradient': [HDx, HDy],
        'Sobel': [HSx, HSy],
        'Prewitt': [HPx, HPy],
        'ISobel': [HISx, HISy]
    }.get(Filter, [HSx, HSy])
    # Filter the image in x- and y-direction 
    IDx = filter_image_float(imgINT, Hx)
    IDy = filter_image_float(imgINT, Hy)
    # create intensity maps and phase maps: 
    E = (np.sqrt(np.multiply(IDx, IDx)+np.multiply(IDy, IDy))).astype(np.float64)
    Phi = (np.arctan2(IDy, IDx)*180/np.pi).astype(np.float64)
    return E, Phi, IDx, IDy

def laplace_sharpen(imgINT, w=0.1, Filter='L4', Threshold=False, TVal=0.1):
    HL = {
        'L4': HL4,
        'L8': HL8,
        'L12': HL12,
    }.get(Filter, HL4)
    edges = filter_image_float(imgINT.astype(np.float64), HL.astype(np.float64))
    edges = np.divide(edges, np.amax(np.abs(edges)))
    if Threshold:
        edges[np.abs(edges) <= TVal] = 0.0
    filtered = (crop_levels(imgINT.astype(np.float) - w * edges.astype(np.float))).astype(np.uint8)
    return filtered, edges

def unsharp_mask(imgINT, a = 4, sigma=2.0, tc = 120.):
    #create a gaussian filter: 
    fsize = (np.ceil(5*sigma) // 2 * 2 + 1) # (5 * next odd integer!)
    HG, HGx, HGy, SUM = create_gaussian_filter(fsize, sigma)

    # filter the image with the Gaussian: 
    imgGaussXY = filter_image_float(filter_image_float(imgINT, HGx), HGy)
    M = imgINT.astype(np.float)-imgGaussXY

    # Create an Edge Map: 
    E, Phi, IDX, IDY = detect_edges(imgINT)
    
    # Threshold our mask with the Edgemap: 
    M[abs(E) <= tc]=0.0

    return (crop_levels(imgINT+a*M.astype(np.float))).astype(np.uint8)



# ----------------------------------------------------------------------------
# Thinning
# ----------------------------------------------------------------------------

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def Thinning(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned


# ----------------------------------------------------------------------------
# HOUGH TRANSFORM 
# ----------------------------------------------------------------------------

# ------------------------------------
def plot_line_rth(M, N, r, theta, ax):
    """
    Plots a line with (r, theta) parametrisation over a given image.
    """
    uc, vc = np.floor_divide(M,2), np.floor_divide(N,2)
    u = np.arange(M)
    v = np.arange(N)
    ax.axis([0, M, 0, N])
    if theta == 0 or theta == np.pi:
        ax.axvline(x = r + uc)
#        print('zero theta line')
    else:
        x = u-uc
        # y = v-vc        
        # m = np.tan(np.pi/2.-theta)
        m = -np.tan(np.pi/2-theta)
        k = r/np.sin(theta)    
        ax.plot(x + uc, -(m * x + k) + vc) 
        
        
# ------------------------------------    
def largest_indices(ARRAY, n):
    """
    Returns the n index combinations referring to the 
    largest values in a numpy array.
    """
    flat = ARRAY.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ARRAY.shape)


# ------------------------------------
def hough_lines_loops(imgBIN, Nth, Nr, K):
    '''
    Computes the Hough transform to detect straight lines in the binary 
    image I (of size M × N ), using Nθ , Nr discrete steps for the angle 
    and radius, respectively. Returns the list of parameter pairs ⟨θi,ri⟩ 
    for the K strongest lines found.
    --- direct implementation of our pseudo code, with if / loops -> SLOW! ---
    '''
    # Find image center: 
    N, M = imgBIN.shape
    uc, vc = np.floor_divide(M,2), np.floor_divide(N,2) # maybe just divide???
    
    # initialise increments:
    rmax = np.sqrt(uc*uc + vc*vc)
    dth = np.pi / Nth
    dr = 2. * rmax / Nr

    # Create the accumulator Array: 
    Acc = np.zeros((Nth, Nr))
    
    # Fill the Accumulator Array:
    for u in range(M):
        for v in range(N):
            if imgBIN[v, u] == 1:
                x, y = u-uc, v-vc
                for ith in range(Nth):
                    theta = dth*ith
                    r = x * np.cos(theta) + y * np.sin(theta)                   
                    ir = np.min([(np.floor_divide(Nr, 2) + np.floor_divide(r,dr)).astype(np.integer), Nr-1])
                    Acc[ith, ir] = Acc[ith, ir] + 1
                    
    MaxIDX = largest_indices(Acc, K)
    MaxTH = dth * MaxIDX[0][:]
    MaxR = dr * MaxIDX[1][:] -rmax
 
    return Acc, MaxIDX, MaxTH, MaxR


def hough_lines(imgBIN, Nth, Nr, K):
    '''
    Computes the Hough transform to detect straight lines in the binary 
    image I (of size M × N ), using Nθ , Nr discrete steps for the angle 
    and radius, respectively. Returns the list of parameter pairs ⟨θi,ri⟩ 
    for the K strongest lines found.
    --- Almost all loops avoided / replaced by array arithmetic 
    -> around 100x faster that the "direct" implementation ---
    '''
    # Find image center: 
    N, M = imgBIN.shape
    uc, vc = np.floor_divide(M,2), np.floor_divide(N,2) # maybe just divide???
    
    # initialise increments:
    rmax = np.sqrt(uc*uc + vc*vc)
    dth = np.pi / Nth
    dr = 2. * rmax / Nr

    # Create the accumulator Array: 
    Acc = np.zeros((Nth, Nr))
    
    # Fill the Accumulator Array:
    # we can avoid the if statement by numpy's "nonzero" function:
    nzi = np.nonzero(imgBIN)
    y = nzi[0][:] - vc
    x = nzi[1][:] - uc
    
    # we can avoid the theta / r loop(s) by 
    # .... 1.) an outer product:
    ith = range(Nth) 
    theta = dth * ith
    r = np.outer(x,np.cos(theta)) + np.outer(y,np.sin(theta))
    ir = (np.floor_divide(Nr, 2) + np.floor_divide(r,dr)).astype(np.integer)
    # (column index = theta index, the rows represent ALL possible r_index values for one theta value)

    # ... and 2.) a column-wise sum of equal radius values: 
    # (I could not find a way to get rid of the "last" remaining loop over r:)
    for cnt in range(Nr):
        Acc[:, cnt] = np.sum(ir == cnt, axis=0)
                    
    MaxIDX = largest_indices(Acc, K)
    MaxTH = dth * MaxIDX[0][:]
    MaxR = dr * MaxIDX[1][:] -rmax

    return Acc, MaxIDX, MaxTH, MaxR

# ------------------------------------
# ------------------------------------
# PLOTTING: 
# ------------------------------------
# ------------------------------------

def plot_hist(I, title='Histogram', color='tab:blue'):
    ''' manual histogram plot (for uint8 coded intensity images) ''' 
    fig, ax = plt.subplots()
    ax.set_xlabel('intensity level')
    ax.set_ylabel('number of pixels', color=color) 
    plth = ax.stem(hist256(I), color, markerfmt=' ', basefmt=' ')
    ax.set_title(title)
    return fig, ax, plth

def plot_cumhist(I, title='Cummulative Histogram', color='tab:red'):
    ''' manual cumulative histogram plot (for uint8 coded intensity images) '''
    fig, ax = plt.subplots()
    ax.set_xlabel('intensity level')
    ax.set_ylabel('cumulative n.o.p.', color=color)
    pltch = ax.plot(cum_hist256(I), color=color, linestyle=' ', marker='.')
    ax.set_title(title)
    return fig, ax, pltch

# COMBINED PLOTTING: 

def plot_image_hist_cumhist(I, title='Intensity Image', cmap='gray', vmax=255, vmin=0):
    ''' function for the combined plotting of the intensity image, its histogram
    and the cumulative histogram. The histograms are wrapped in a single plot, 
    but since the scales are different, we introduce two y axes (left and right, 
    with different color).'''
    
    # plot the intensity image: 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plti = ax1.imshow(I, cmap=cmap, vmax=vmax, vmin=vmin)
    ax1.set_title(title)
    fig.colorbar(plti, ax=ax1)
    
    # plot the histograms in a yy plot: 
    color = 'tab:blue'
    ax2.set_xlabel('intensity level')
    ax2.set_ylabel('number of pixels', color=color) 
    plth = ax2.stem(hist256(I), color, markerfmt=' ', basefmt=' ')
    ax2.tick_params(axis='y', labelcolor=color) 
    ax2.set_title('Histogram')

    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    pltch = ax3.plot(cum_hist256(I), color=color, linestyle=' ', marker='.')
    ax3.set_ylabel('cumulative n.o.p.', color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    
    plt.tight_layout()
    
    return fig, ax1, ax2, ax3, plti, plth, pltch

def plot_image_all_hists(img, title='Combined Histograms', cmap="gray", vmax=255, vmin=0):
    ''' combined function for plotting the intensity image next to
    * all component histograms, 
    * the intensity histogram and the 
    * cumulative histogram,
    all wrapped in a single plot'''
    
    # separate and combine
    imgRED = img[:,:,0]
    imgGREEN = img[:,:,1]
    imgBLUE = img[:,:,2]
    imgLUM = rgb2GrayLuminosity(img)
    
    # plot the intensity image: 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plti = ax1.imshow(imgLUM, cmap=cmap, vmax=vmax, vmin=vmin)
    ax1.set_title(title)
    fig.colorbar(plti, ax=ax1)
    
    # plot the histograms in a yy plot: 
    color = 'black'
    ax2.set_xlabel('intensity level')
    ax2.set_ylabel('number of pixels', color=color)
    numBins=256
    pltLUM = ax2.hist(imgLUM.flatten(), numBins, color='black')
    pltR = ax2.hist(imgRED.flatten(), numBins, color='tab:red', alpha=0.5)
    pltG = ax2.hist(imgGREEN.flatten(), numBins, color='tab:green', alpha=0.5)
    pltB = ax2.hist(imgBLUE.flatten(), numBins, color='tab:blue', alpha=0.5)
    

    ax2.tick_params(axis='y', labelcolor=color) 
    ax2.set_title('Histogram')
    
    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    pltch = ax3.plot(cum_hist256(imgLUM), color=color, linestyle=' ', marker='.')
    ax3.set_ylabel('cumulative n.o.p.', color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    
    plt.tight_layout()
    
    return fig, ax1, ax2, ax3, plti, pltLUM, pltch

def plot_image_sequence(sequence, title='Intensity Image', cmap='gray', vmax=255, vmin=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    for i in range(sequence.shape[0]):
        ax.imshow(sequence[i,:,:], cmap=cmap, vmax=vmax, vmin=vmin)
        ax.set_title('threshold level: %3i' % (i))
        plt.pause(0.01)
    return fig, ax