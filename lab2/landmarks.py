import cv2
import os
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import procrustes

landmarks = pd.read_csv("dataset_chicago/landmarks/landmark_templates_01-29.22/Template Database CSV 012922.csv")

image_names = landmarks['fname'].unique()

# Create a list to store concatenated coordinates for each image
concatenated_coordinates_list = []

# Iterate over each image
for image_name in image_names:
    # Select rows corresponding to the current image
    image_landmarks = landmarks[landmarks['fname'] == image_name]
    
    # Extract x and y coordinates
    x_coordinates = image_landmarks['x'].values
    y_coordinates = image_landmarks['y'].values
    
    # Concatenate x and y coordinates
    concatenated_coordinates = np.concatenate([x_coordinates, y_coordinates])
    
    # Append to the list
    concatenated_coordinates_list.append(concatenated_coordinates)

# Convert the list to a NumPy array
landmarks = np.array(concatenated_coordinates_list).T
print(landmarks)
print(landmarks.shape)

num_faces = landmarks.shape[1]
num_landmarks = landmarks.shape[0]//2

#A PARTIR DAQUI SHA DE MIRAR COM FER PROCRUSTERS

# Reshape the landmarks to (num_faces, num_landmarks, 2)
landmarks_reshaped = landmarks.reshape((num_faces, num_landmarks, 2))

# Choose one face as the reference
reference_landmarks = landmarks_reshaped[0]

# Align/center all faces using Procrustes analysis
aligned_landmarks = np.zeros_like(landmarks_reshaped)
for i in range(num_faces):
    _, _, aligned_landmarks[i] = procrustes(reference_landmarks, landmarks_reshaped[i])

# Reshape aligned landmarks back to (num_faces, num_landmarks * 2)
aligned_landmarks_flat = aligned_landmarks.reshape((num_faces, -1))

for i in range(5):
    # Plot original landmarks
    plt.scatter(landmarks_reshaped[i, :, 0], landmarks_reshaped[i, :, 1], label='Original', marker='o')
    
    # Plot aligned landmarks
    plt.scatter(aligned_landmarks[i, :, 0], aligned_landmarks[i, :, 1], label='Aligned', marker='x')
    
    plt.title(f"Landmarks for Face {i+1}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.show()

'''
for i in range(5): # visualize the landmarks for each face
    landmarks[:, 1, i] = -landmarks[:, 1, i]
    plt.scatter(landmarks[:, 0, i], landmarks[:, 1, i])
    plt.title(f"Landmarks for Face {i+1}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()'''

'''
# Reshape aligned landmarks to a 2D array for further processing
num_landmarks, num_coordinates, num_samples = aligned_landmarks.shape
reshaped_landmarks = aligned_landmarks.reshape((num_landmarks * num_coordinates, num_samples)).T

# Subtract the mean landmark coordinates from each individual face's landmarks
mean_landmark = np.mean(reshaped_landmarks, axis=0)
subtracted_landmarks = reshaped_landmarks - mean_landmark

# Vectorize the subtracted landmarks
vectorized_landmarks = subtracted_landmarks.transpose()

# Compute pseudo-covariance matrix L = Xt*X
L_landmarks = np.dot(vectorized_landmarks.T, vectorized_landmarks)

# Compute eigenvalues and eigenvectors
eigenvalues_landmarks, eigenvectors_landmarks = np.linalg.eigh(L_landmarks)

# Keep only the positive eigenvalues and their corresponding eigenvectors
positive_eigenvalue_indices_landmarks = eigenvalues_landmarks > 0
print('positive eigenvalues', positive_eigenvalue_indices_landmarks.shape)
eigenvalues_landmarks = eigenvalues_landmarks[positive_eigenvalue_indices_landmarks]
eigenvectors_landmarks = eigenvectors_landmarks[:, positive_eigenvalue_indices_landmarks]

# Sort eigenvalues and corresponding eigenvectors
sort_indices_landmarks = np.argsort(eigenvalues_landmarks)[::-1]
eigenvalues_landmarks = eigenvalues_landmarks[sort_indices_landmarks]
eigenvectors_landmarks = eigenvectors_landmarks[:, sort_indices_landmarks]

# Normalize the eigenvectors
normalized_eigenvectors_landmarks = eigenvectors_landmarks / np.linalg.norm(eigenvectors_landmarks, axis=0)
normalized_eigenvalues_landmarks = eigenvalues_landmarks / sum(eigenvalues_landmarks)

# Print shapes of eigenvalues and eigenvectors
print("Eigenvalues (Landmarks):", eigenvalues_landmarks.shape)
print("Eigenvectors (Landmarks):", eigenvectors_landmarks.shape)

# Number of principal components to keep
components_landmarks = 10

# Visualization of eigenvalues for landmarks
plt.bar(range(1, components_landmarks + 1), normalized_eigenvalues_landmarks[:components_landmarks])
plt.title("Eigenvalues of Aligned Landmarks")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Eigenvalue Magnitude")
plt.show()

# Additional analysis to determine the meaningfulness of the extracted bases can be done here.
# Choose a criterion to validate the 10 extracted bases, as mentioned in the project requirements.


'''