import cv2
import os
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import procrustes

landmarks = pd.read_csv("dataset_chicago/landmarks/landmark_templates_01-29.22/Template Database CSV 012922.csv")
landmarks['fname'] = landmarks['fname'].str.replace('.tem', '')

faces_landmarks = [] # we will create a (189, 2, 597) array

# all_landmarks = landmarks[['x', 'y']].values # have all landmarks together sense tenir en compte a quina cara pertanyen cada un ??

for filename in landmarks['fname'].unique():
    subset = landmarks[landmarks['fname'] == filename]
    faces_landmarks.append(subset[['x', 'y']].values)

landmark_coordinates = np.array(faces_landmarks)
arr_transposed = np.transpose(landmark_coordinates, (1, 2, 0))
landmarks = arr_transposed # shape (189, 2, 597) --> una matriu de 189 rows (1 per cada landmark) i 2 columns (x, y coordinates), per cada una de les 597 samples
# nose si directament podem agafar tots els punts junts sense tenir en compte quins punts pertanyen a cada cara

reference_landmarks = landmarks[:, :, 0]  # Choose one face as the reference
aligned_landmarks = np.zeros_like(landmarks)

# perform procruster to align/center all faces
for i in range(597):
    _, _, aligned_landmarks[:, :, i] = procrustes(reference_landmarks, landmarks[:, :, i])


for i in range(597): # visualize the landmarks for each face
    landmarks[:, 1, i] = -landmarks[:, 1, i]
    plt.scatter(landmarks[:, 0, i], landmarks[:, 1, i])
    plt.title(f"Landmarks for Face {i+1}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()


