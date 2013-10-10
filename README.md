VizardCurvesTool
================

import viz
import math
import random
import vizact
import vizspace
import vizcam
import vizinfo
import numpy

viz.cam.setHandler(vizcam.KeyboardCamera())
viz.setMultiSample(4)
viz.fov(60)
viz.go()
	
#viz.addChild('ground.osgb')
#pic = viz.addTexture('87.png')
## Create surface to wrap the texture on
#quad = viz.addTexQuad()
##put quad in view
#quad.setPosition([0, 1.5, 4])
## Wrap texture on quad
#quad.texture(pic)

def load_image(file_path):
	img_array = scipy.misc.imread(file_path) 
	return img_array

#def plot_image(img_array):
#	plt.imshow(img_array, cmap = 'gray')

def rgb2gray(img_rgb):
	R_CONST = 0.299
	G_CONST = 0.587
	B_CONST = 0.114
	img_shape = img_rgb.shape
	M = img_shape[0]
	N = img_shape[1]
	K = img_shape[2]
	#declare empty array for grayscale values
	img_gray = np.zeros([M,N])
	for m in range(M):
		for n in range(N):
			img_rgb_r = img_rgb[m,n,0]
			img_rgb_b = img_rgb[m,n,1]
			img_rgb_g = img_rgb[m,n,2]
			img_gray[m,n] = (R_CONST * img_rgb_r) + (G_CONST * img_rgb_g) + (B_CONST * img_rgb_b)
	return img_gray

def hist_eq(img_hist, reduction_factor = 8):
	#use cumulative distribution function
	cdf = cumulative_sum(img_hist)
	cdf = 255 * cdf / cdf[-1]
	#linearly interpolate the cdf for equalized pixel values
	img_equalized = np.interp(img.flatten(), np.arange(256), cdf)
	#reshape to a matrix with dimensions of the original image
	img_equalized = img_equalized.reshape(img.shape)
	return img_equalized, cdf

def hist_256(img_array):
	#initialize histogram vector
	hist_vector = np.zeros(256)
	for i in range(img_array.shape[0]):
		for j in range(img_array.shape[1]):
			pixel_val = int(round(img_array[i,j]))
			hist_vector[pixel_val] = hist_vector[pixel_val] + 1
	return hist_vector
	
def cumulative_sum(array):
	#returns an array of cumulative sums for the value in a 1D array
	array_length = array.shape[0]
	cumulative_sum = np.zeros(array_length)
	for i in range(array_length):
		if i == 0:
			cumulative_sum[i] = array[i]
		else:
			cumulative_sum[i] = cumulative_sum[i-1] + array[i]
	return cumulative_sum
	
def shiftCurves(image, slidervalue):
	image = np.array.flatten(image)
	image_length = image.shape[0]
	grey = rgb2gray(image)
	grey_hist = hist_256(grey)
	slidervalue = slidervalue/(array_length/2)
	count = 127;
	for i in range(grey_hist):
		if (slidervalue*i <= slidervalue):
			grey_hist[i] += slidervalue*i
		else:
			grey_hist[i] += slidervalue*i - slidervalue * (i - count)
			count -= 1
	img_eq, cdf = hist_eq(grey_hist, 8)	
	return img_eq, cdf
	
if __name__ == "_main_":
	img = load_image('87.png')
	modimage, cdf = shiftCurves(img, 20)
	quadTexture = viz.addTexture(modimage)
	# Create surface to wrap the texture on
	quad = viz.addTexQuad()
	quad.texture(quadTexture)
	#put quad in view
	quad.setPosition([0, 1.5, 4])
	# Wrap texture on quad
	quad.texture(pic)
