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


for i in range(5): # visualize the landmarks for each face
    landmarks[:, 1, i] = -landmarks[:, 1, i]
    plt.scatter(landmarks[:, 0, i], landmarks[:, 1, i])
    plt.title(f"Landmarks for Face {i+1}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()


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


