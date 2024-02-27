import os
import numpy as np
from imageio import imread
#from imageio.v2 import imread
from scipy.io import loadmat
import pandas as pd
import random
import time
import itertools
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    # transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((96, 96)),  # Resize the image
    transforms.ToTensor()  # Convert image to tensor
])

def my_face_recognition_function(A, my_FRmodel):
    # Convert the cropped image to PIL Image
    pil_image = Image.fromarray(A)

    '''# Check the number of channels in the input image
    if pil_image.mode == 'L':
        # Grayscale image (1 channel) - convert to RGB
        pil_image = pil_image.convert('RGB')'''

    # Apply transformations
    transformed_image = transform(pil_image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        # Forward pass through the model
        logits = my_FRmodel(transformed_image)

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)

        # Get the predicted class (identity)
        _, predicted_class = torch.max(probabilities, 1)

        # Convert to numpy array and extract the predicted class index
        predicted_class_index = predicted_class.numpy()[0]

    return predicted_class_index, probabilities

def grid_search_threshold(model, images, true_labels):
    thresholds = np.linspace(0, 1, 100)  # Define thresholds to search over
    best_threshold = 0
    best_score = 0

    for threshold in thresholds:
        AutoRecognSTR = []
        for idx, image in enumerate(images):
            # Perform face recognition with the given threshold
            predicted_class_index, probabilities = my_face_recognition_function(image, model)
            if torch.max(probabilities) < threshold:
                auto_id = -1  # Assign -1 if probability is under the threshold
            else:
                auto_id = predicted_class_index + 1
            AutoRecognSTR.append(auto_id)

        # Compute F1-score for the current threshold
        FR_score = CHALL_AGC_ComputeRecognScores(AutoRecognSTR, true_labels)
        if FR_score > best_score:
            best_score = FR_score
            best_threshold = threshold

    return best_threshold

# Perform grid search for the optimal threshold
optimal_threshold = grid_search_threshold(my_FRmodel, cropped_images, ids)

# Use the optimal threshold for face recognition
predicted_class_index, probabilities = my_face_recognition_function(cropped_images[count], my_FRmodel, threshold=optimal_threshold)