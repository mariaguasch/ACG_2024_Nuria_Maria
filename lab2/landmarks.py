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
num_faces = 597
num_landmarks = 189

# Create a list to store concatenated coordinates for each image
concatenated_coordinates_list = []

faces_landmarks = [] # my_list

# Iterate over each image
for image_name in image_names:
    # Select rows corresponding to the current image
    image_landmarks = landmarks[landmarks['fname'] == image_name]
    faces_landmarks.append(image_landmarks[['x', 'y']].values) #
    
    # Extract x and y coordinates
    x_coordinates = image_landmarks['x'].values
    y_coordinates = image_landmarks['y'].values
    
    # Concatenate x and y coordinates
    concatenated_coordinates = np.concatenate([x_coordinates, y_coordinates])
    
    # Append to the list
    concatenated_coordinates_list.append(concatenated_coordinates)

'''# Convert the list to a NumPy array
landmarks = np.array(concatenated_coordinates_list).T
print(landmarks)
print(landmarks.shape)'''

####
landmark_coordinates = np.array(faces_landmarks)
arr_transposed = np.transpose(landmark_coordinates, (1, 2, 0))
landmarks = arr_transposed #shape (189, 2, 597)
print('landmarks:', landmarks.shape)
####

reference_landmarks = landmarks[:, :, 0]  # Choose one face as the reference
aligned_landmarks = np.zeros_like(landmarks)

# perform procruster to align/center all faces
for i in range(597):
    _, aligned_landmarks[:, :, i], m = procrustes(reference_landmarks, landmarks[:, :, i])

vect_landmarks = [] # matrix containing a column for each face

for i in range(597): 
    
    # visualize the landmarks for some example faces
    if i in [0, 1]:
        # aligned_landmarks[:, 1, i] = -aligned_landmarks[:, 1, i] # to undo the flip around the x-axis
        plt.scatter(aligned_landmarks[:, 0, i], - aligned_landmarks[:, 1, i])
        plt.title(f"Landmarks for Face {i+1}")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.show()

    # convert coordinates to matrix of column vectors
    x_coords = aligned_landmarks[:, 0, i]
    y_coords = - aligned_landmarks[:, 1, i] # we add the - sign to have the faces upright
    coords_array = np.concatenate([x_coords, y_coords])
    vect_landmarks.append(coords_array)

vect_landmarks = np.array(vect_landmarks).transpose()
print(vect_landmarks.shape)

# Subtract the mean landmark coordinates from each individual face's landmarks
mean_landmark = np.mean(vect_landmarks, axis=1)
subtracted_landmarks = vect_landmarks - mean_landmark[:, np.newaxis]
print('substracted_landmarks:', subtracted_landmarks.shape)

# Compute pseudo-covariance matrix L = Xt*X
L_landmarks = np.dot(subtracted_landmarks, subtracted_landmarks.T)

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
print("Eigenvalues (Landmarks):", normalized_eigenvalues_landmarks.shape)
print("Eigenvectors (Landmarks):", normalized_eigenvectors_landmarks.shape)

components_landmarks = 10 # number of components to keep

##### VISUALIZATION OF EIGENFACES ####
count = 0
plt.close()

for eig_face in normalized_eigenvectors_landmarks.transpose():
    plt.title(f"Eigenface{count + 1}")
    plt.scatter(eig_face[:189], eig_face[189:], color = 'blue')
    plt.scatter(aligned_landmarks[:, 0, 0], - aligned_landmarks[:, 1, 0], color = 'red')
    plt.axis("off")
    plt.show()
    count +=1
    if count == components_landmarks: break

######################################

plt.close()

# Visualization of eigenvalues for landmarks
plt.bar(range(1, components_landmarks + 1), normalized_eigenvalues_landmarks[:components_landmarks])
plt.title("Eigenvalues of Aligned Landmarks")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Ratio of variance (normalized eigenvalue)")
plt.show()

projected_images = np.dot(vect_landmarks.T, normalized_eigenvectors_landmarks[:, :components_landmarks])
reconstructed_images = np.dot(projected_images, normalized_eigenvectors_landmarks[:, :components_landmarks].T).transpose() + mean_landmark[:, np.newaxis]
print('reconstructed shape', reconstructed_images.shape)

for i in range(3): 
    plt.scatter(reconstructed_images[:189, i], reconstructed_images[189:, i])
    plt.title(f"Reconstructed {i+1}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()

# Additional analysis to determine the meaningfulness of the extracted bases can be done here.
# Choose a criterion to validate the 10 extracted bases, as mentioned in the project requirements.


# MODES OF VARIATION
num_faces_to_visualize = 3

fig, axes = plt.subplots(nrows=num_faces_to_visualize, ncols=components_landmarks, figsize=(15, 5))

for i in range(components_landmarks):
    for j in range(num_faces_to_visualize):
        
        # do reconstruction with increasing subset of basis
        reconstructed_face = mean_landmark + np.dot(projected_images[j, :i+1], normalized_eigenvectors_landmarks[:, :i+1].T)
        
        reconstructed_face = reconstructed_face.reshape(2, num_landmarks).T
    
        axes[j, i].scatter(reconstructed_face[:, 0], reconstructed_face[:, 1])
        axes[j, i].set_title(f"Basis {i+1}")
        axes[j, i].axis("off")

plt.tight_layout()
plt.show()
