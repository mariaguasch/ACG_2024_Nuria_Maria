import numpy as np
from imageio import imread, imwrite
import pandas as pd
import time
import os

import cv2 as cv

def MyFaceDetectionFunction(image, faces_not_detected_count):
    # Function to implement
    haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    nose_cascade = cv.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')    
    
    if len(image.shape) == 3:  # Check if the image is in color
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        # Handle case where the image is already grayscale
        grayscale = image

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
    
    # Sort valid faces based on the number of features detected
    valid_faces = sorted(valid_faces, key=lambda x: x[4], reverse=True)

    if len(valid_faces) > 0:
        valid_face = valid_faces[0]  # only keep the largest face
        return valid_face, faces_not_detected_count
    else:
        faces_not_detected_count += 1
        return None, faces_not_detected_count

# Provide the path to the directory containing subdirectories with images
main_directory = "/home/maria/Documentos/GitHub/ACG_2024_Nuria_Maria/lab4/Python/photos"

# Specify the folder to save all cropped images
save_folder = "/home/maria/Documentos/GitHub/ACG_2024_Nuria_Maria/lab4"

faces_not_detected_count = 0

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

            # Load the color image
            A = imread(img_path)

            # Perform face detection
            det_face, faces_not_detected_count = MyFaceDetectionFunction(A, faces_not_detected_count)

            if det_face is not None:
                # Extract face coordinates
                x, y, x2, y2, _ = det_face

                # Crop the original color image using the face coordinates
                cropped_image = A[y:y2, x:x2]

                # Save the cropped image in the specified folder with the same name as the original image
                save_path = os.path.join(save_folder, "new_cropped_images")
                os.makedirs(save_path, exist_ok=True)
                filename = os.path.join(save_path, f"cropped_{os.path.basename(img_path)}")
                imwrite(filename, cropped_image)
                # print(f"Cropped image saved: {filename}")

print('Faces not detected =', faces_not_detected_count)
