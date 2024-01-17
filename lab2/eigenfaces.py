from numpy import load
import numpy as np
from imageio import imread
from scipy.io import loadmat
import pandas as pd
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import random

# import cv2 as cv

# npz_file = np.load(r'C:\Users\guasc\Documentos\GitHub\ACG_2024_Nuria_Maria\lab2\dataset\face_images.npz')

# npz_file = np.load('your_file.npz')
# Access the images (assuming they are stored as an array with a specific key, e.g., 'images')
# images_array = npz_file['face_images']

image_dir = r'C:\Users\guasc\Documentos\GitHub\ACG_2024_Nuria_Maria\lab2\dataset\face_images.npz'
images = np.load(image_dir)['face_images']
# Standardize the values of images 
images = images/255.0

# Plot a random image
m = images.shape[2] # Number of samples
print(m)

for i in range(m):
    plt.imshow(images[:,:,i], cmap='gray')
    plt.show()

# # Plot a random image
# index = random.randint(0, m-1)
# plt.imshow(images[:,:,index], cmap='gray')
# plt.show()