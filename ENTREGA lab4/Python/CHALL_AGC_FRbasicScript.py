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

#####################################################################

# Define model architecture
class ReducedIdEstimationModel(nn.Module):
    #Python/reduced_80epochs_batch400_resize150_conv64.ckpt
    #Python/reduced_100epochs_batch400_resize150_conv64.ckpt
    def __init__(self, num_classes):
        super(ReducedIdEstimationModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Removed one Conv2d layer to reduce parameters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), 
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 9 * 9, 128),  # Corrected input size based on the last layer's output
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((150, 150)),  # Resize the image to 150x150
    transforms.ToTensor()  # Convert image to tensor
])


def CHALL_AGC_ComputeRecognScores(auto_ids, true_ids):
    #   Compute face recognition score
    #
    #   INPUTS
    #     - AutomSTR: The results of the automatic face
    #     recognition algorithm, stored as an integer
    #
    #     - AGC_Challenge_STR: The ground truth ids
    #
    #   OUTPUT
    #     - FR_score:     The final recognition score
    #
    #   --------------------------------------------------------------------
    #   AGC Challenge
    #   Universitat Pompeu Fabra
    #

    if len(auto_ids) != len(true_ids):
        assert ('Inputs must be of the same len')

    f_beta = 1
    res_list = list(filter(lambda x: true_ids[x] != -1, range(len(true_ids))))

    nTP = len([i for i in res_list if auto_ids[i] == true_ids[i]])

    res_list = list(filter(lambda x: auto_ids[x] != -1, range(len(auto_ids))))

    nFP = len([i for i in res_list if auto_ids[i] != true_ids[i]])

    res_list_auto_ids = list(filter(lambda x: auto_ids[x] == -1, range(len(auto_ids))))
    res_list_true_ids = list(filter(lambda x: true_ids[x] != -1, range(len(true_ids))))

    nFN = len(set(res_list_auto_ids).intersection(res_list_true_ids))

    FR_score = (1 + f_beta ** 2) * nTP / ((1 + f_beta ** 2) * nTP + f_beta ** 2 * nFN + nFP)

    return FR_score


def my_face_recognition_function(A, my_FRmodel):
# Convert the cropped image to PIL Image
    pil_image = Image.fromarray(A)
    
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

def my_face_detection(grayscale, name):
    # Function to implement
    haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    nose_cascade = cv.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')    
    
    detected_faces = haar_cascade.detectMultiScale(grayscale, scaleFactor=1.2, minNeighbors=3)

    valid_faces = []

    # Iterate through all the detected faces, to then use other classifiers to get better accuracy
    for (x, y, w, h) in detected_faces: 
        face_roi = grayscale[y:y+h, x:x+w]
        detected_eyes = eye_cascade.detectMultiScale(face_roi)
        detected_nose = nose_cascade.detectMultiScale(face_roi)
        detected_mouth = mouth_cascade.detectMultiScale(face_roi)
    
        all_detected =  len(detected_eyes) + len(detected_mouth) + len(detected_nose)
        
        # We have created a "score" which tells us how many features have been detected in a face 
        # If at least 3 of them are, we consider this face as valid

        if all_detected >= 3:
            valid_faces.append([int(x), int(y), int(x + w), int(y + h), all_detected])

    valid_faces = sorted(valid_faces, key=lambda x: x[4], reverse=True)
    # We only keep the top 2 faces with highest score, meaning that more of its features (eyes, mouth, nose) have been detected

    valid_faces = valid_faces[:2]

    cropped_images = []
    for face in valid_faces:
        cropped_image = grayscale[face[1]:face[3], face[0]:face[2]]
        cropped_images.append(cropped_image)


    valid_faces = [[x[0], x[1], x[2], x[3]] for x in valid_faces]

    return valid_faces, cropped_images


# Basic script for Face Recognition Challenge
# ----------------------------------------------------------------------------------------------------------------------------------
# AGC Challenge
# Universitat Pompeu Fabra

# Load challenge Training data
dir_challenge3 = " "
AGC_Challenge3_TRAINING = loadmat('/home/maria/Documentos/GitHub/ACG_2024_Nuria_Maria/lab4/AGC_Challenge3_Training.mat')  # Replace with your path !!!
AGC_Challenge3_TRAINING = np.squeeze(AGC_Challenge3_TRAINING['AGC_Challenge3_TRAINING'])

imageName = AGC_Challenge3_TRAINING['imageName']
imageName = list(itertools.chain.from_iterable(imageName))

ids = list(AGC_Challenge3_TRAINING['id'])
ids = np.concatenate(ids).ravel().tolist()

faceBox = AGC_Challenge3_TRAINING['faceBox']
faceBox = list(itertools.chain.from_iterable(faceBox))

imgPath = "/home/maria/Documentos/GitHub/ACG_2024_Nuria_Maria/lab4/TRAINING/"  # Replace with your path !!!

# Initialize results structure
AutoRecognSTR = []

# Initialize timer accumulator
total_time = 0

# Load your FRModel
my_FRmodel = ReducedIdEstimationModel(num_classes=80)

my_FRmodel.load_state_dict(torch.load('Python/reduced_100epochs_batch400_resize150_conv64.ckpt', map_location=torch.device('cpu')))  # Replace with your path !!!
my_FRmodel.eval()

print('Iterating through the images...')

for idx, im in enumerate(imageName):

    A = imread(imgPath + im)

    try:
        ti = time.time()
        # Timer on
        ###############################################################
        # Your face recognition function goes here.It must accept 2 input parameters:
        #1. FACE DETECTION
        #2. FACE RECOGNITION

        # 1. the input image A
        # 2. the recognition model

        # and must return a single integer number as output, which can be:

        # a) A number between 1 and 80 (representing one of the identities in the training set)
        # b) A "-1" indicating that none of the 80 users is present in the input image


        #STEP 1 -> FACE DETECTION (lab 1)
        if not len(A.shape) == 2:
            grayscale = cv.cvtColor(A, cv.COLOR_BGR2GRAY)
        else:
            # Handle case where image is already grayscale
            grayscale = A

        det_faces, cropped_images = my_face_detection(grayscale, im)

        if len(det_faces) == 0: # if no face is detected in the image, directly return -1
            autom_id = -1
            AutoRecognSTR.append(autom_id)
            continue

        #STEP 2 -> FACE RECOGNITION WITH TRAINED MODEL 

        our_ids = []

        for count, image in enumerate(det_faces):
            predicted_class_index, probabilities = my_face_recognition_function(cropped_images[count], my_FRmodel)

            our_ids.append(predicted_class_index + 1)

        #As There are no images with more than one user in them, only a single identity value must be returned for each image -> we return max id. 
        autom_id = max(our_ids)
        if idx%50 == 0:
            print('Number of processed images:', idx, '/', len(imageName))
        
        tt = time.time() - ti
        total_time = total_time + tt
    except:
        # If the face recognition function fails, it will be assumed that no user was detected for his input image
        print("I'm here for image", im)
        autom_id = random.randint(-1, 80)

    AutoRecognSTR.append(autom_id)

FR_score = CHALL_AGC_ComputeRecognScores(AutoRecognSTR, ids)
_, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print('F1-score: %.2f, Total time: %2d m %.2f s' % (100 * FR_score, int(minutes), seconds))
print('Number of parameters:', sum(p.numel() for p in my_FRmodel.parameters()))
