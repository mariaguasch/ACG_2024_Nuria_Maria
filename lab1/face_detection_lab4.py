import numpy as np
from imageio import imread
import pandas as pd
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os

import cv2 as cv

def MyFaceDetectionFunction(grayscale):
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
    
        all_detected = len(detected_eyes) + len(detected_mouth) + len(detected_nose)
        
        # We have created a "score" which tells us how many features have been detected in a face 
        # If at least 3 of them are, we consider this face as valid

        if all_detected >= 3:
            valid_faces.append([int(x), int(y), int(x + w), int(y + h), all_detected])
            # agafar la greyscale retallar i guardar en una carpeta

    return valid_faces

# Provide the path to the input images directory
imgPath = "/home/maria/Documentos/GitHub/ACG_2024_Nuria_Maria/lab4/Python/photos/Alec Baldwin"
imgFiles = [os.path.join(imgPath, file) for file in os.listdir(imgPath) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Initialize results structure
DetectionSTR = []

# Initialize timer accumulator
total_time = 0

# Iterate through the image filenames
for idx, im in enumerate(imgFiles):
    A = imread(im)
    try:
        ti = time.time()
        # Timer on
        ###############################################################
        # Your face detection function goes here. It must accept a single
        # input parameter (the input image A) and it must return one or
        # more bounding boxes corresponding to the facial images found
        # in image A, specified as [x1 y1 x2 y2]
        # Each bounding box that is detected will be indicated in a
        # separate row in det_faces

        if not len(A.shape) == 2:
            grayscale = cv.cvtColor(A, cv.COLOR_BGR2GRAY)
        else:
            # Handle case where image is already grayscale
            grayscale = A

        det_faces = MyFaceDetectionFunction(grayscale)

        tt = time.time() - ti
        total_time = total_time + tt

        # Plotting the image with detected faces
        fig, ax = plt.subplots()
        ax.imshow(A)
        for bbox in det_faces:
            fb = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(fb)

        plt.title('Detected Faces')
        plt.show()
        plt.clf()
        plt.close()

    except Exception as e:
        # If the face detection function fails, it will be assumed that no
        # face was detected for this input image
        print(f"Caught an exception in {im}: {type(e).__name__} - {str(e)}")
        det_faces = []

    DetectionSTR.append(det_faces)

# Print total processing time
_, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print('Total processing time: %2d m %.2f s' % (int(minutes), seconds))
