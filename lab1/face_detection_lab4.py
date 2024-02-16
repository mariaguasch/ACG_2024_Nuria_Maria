import numpy as np
from imageio import imread, imwrite
import pandas as pd
import time
import os

import cv2 as cv

def MyFaceDetectionFunction(grayscale, save_folder, original_filename):
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
    valid_faces = sorted(valid_faces, key=lambda x: x[4], reverse=True)
    # We only keep the top faces with the highest score, meaning that more of its features (eyes, mouth, nose) have been detected

    if valid_faces:
        valid_face = valid_faces[0]  # only keep the largest face
        cropped_image = grayscale[valid_face[1]:valid_face[3], valid_face[0]:valid_face[2]]

        # Save the cropped image in the specified folder with the same name as the original image but with 'cropped_' added at the beginning
        save_path = os.path.join(save_folder, "cropped_images")
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f"cropped_{os.path.basename(original_filename)}")
        imwrite(filename, cropped_image)
        print(f"Cropped image saved: {filename}")

        valid_faces = [[x[0], x[1], x[2], x[3]] for x in valid_faces]
        return valid_face
    else:
        print("No valid faces found.")
        return None

# Provide the path to the directory containing subdirectories with images
main_directory = "/home/maria/Documentos/GitHub/ACG_2024_Nuria_Maria/lab4/Python/photos"

# Specify the folder to save all cropped images
save_folder = "/home/maria/Documentos/GitHub/ACG_2024_Nuria_Maria/lab4/cropped_images"
# HE CANVIAT EL PATH PERQ ES GUARDIN A LA CARPETA DE LAB 4 PERO NO HO HE EXECUTAT, ESTAN DE MOMENT A LA DE LAB 1

# Iterate through subdirectories (folders) in the main directory
for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)
    
    # Check if the item in the directory is a subdirectory
    if os.path.isdir(folder_path):
        print(f"Processing images in folder: {folder}")
        
        # Iterate through the image filenames in the subdirectory
        for img_filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_filename)

            # Skip non-image files like .DS_Store
            if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue

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

                A = imread(img_path)

                if not len(A.shape) == 2:
                    grayscale = cv.cvtColor(A, cv.COLOR_BGR2GRAY)
                else:
                    # Handle case where the image is already grayscale
                    grayscale = A

                det_face = MyFaceDetectionFunction(grayscale, save_folder, img_filename)

                tt = time.time() - ti

            except Exception as e:
                # If the face detection function fails, it will be assumed that no
                # face was detected for this input image
                print(f"Caught an exception in {img_path}: {type(e).__name__} - {str(e)}")
