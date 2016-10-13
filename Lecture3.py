
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im = Image.open('dolphins.jpg')
im = im.convert('L')
plt.gray()

im1 = np.array(im) 


def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """ 
    # get image histogram 
    imhist,bins = np.histogram(im.flatten(), nbr_bins, normed=True) 
    cdf = imhist.cumsum() 
    # cumulative distribution function 
    cdf = 255 * cdf / cdf[-1] 
    # normalize 
    # use linear interpolation of cdf to find new pixel values 
    im2 = np.interp(im.flatten(),bins[:-1],cdf) 
    return im2.reshape(im.shape), cdf, bins 

def hist_match(im, g_cdf, bins):
    output_image = np.interp(im.flatten(),g_cdf,bins[:-1]) 
    return output_image.reshape(im.shape)   


def getFig():

    ax = []

    ax.append(plt.subplot2grid((4,2), (0,0)))
    ax.append(plt.subplot2grid((4,2), (0,1)))
    ax.append(plt.subplot2grid((4,2), (1,0)))
    ax.append(plt.subplot2grid((4,2), (1,1)))
    ax.append(plt.subplot2grid((4,2), (2,0), colspan=2))
    ax.append(plt.subplot2grid((4,2), (3,0)))
    ax.append(plt.subplot2grid((4,2), (3,1)))
    
    return ax

def Task():
    ax = getFig()

    bcount = 256

    # Input image with its histogram.
    ax[0].imshow(im)
    ax[0].axis('off')

    ax[2].hist(im1.flatten(), bcount)
    
    # Random normal distribution.
    normal_distribution = np.random.normal(127, 32, 10000)

    # Desired probability density function.
    ax[4].hist(normal_distribution, bcount)

    # Getting T(x).
    output_uniform, cdf1, bins1 = histeq(im1, bcount)

    # Getting G(x).
    output_random, cdf2, bins2 = histeq(normal_distribution, bcount)

    # Transform function for the input image.
    ax[5].plot(cdf1)
    # Transform function for getting uniform from the normal.
    ax[6].plot(cdf2)

    # Resulting image using matching algorithm.
    output_image = hist_match(output_uniform, cdf2, bins2)
    output_image_data = np.array(output_image)
    ax[1].axis('off')
    ax[1].imshow(output_image)
    
    # Real output image histogram.
    ax[3].hist(output_image_data.flatten(), bcount)

    plt.savefig('ResultImage_L3.jpg')

    plt.show()
    
Task()