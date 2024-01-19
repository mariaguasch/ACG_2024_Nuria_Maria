import cv2
import os
import numpy as np

# Directory containing face images
faces_directory = "dataset_chicago/cfd/CFD Version 3.0/Images/NeutralFaces"
target_height, target_width = 90, 128

# Load all face images
face_images = []
for filename in os.listdir(faces_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        img_path = os.path.join(faces_directory, filename)
        img = cv2.imread(img_path)  # Read images in grayscale
        img_resized = cv2.resize(img, (target_width, target_height))
        face_images.append(img_resized)

# Convert the list of images to a NumPy array
face_images_array = np.array(face_images)
print('Faces converted to array')

# Calculate the mean face
mean_face = np.mean(face_images_array, axis=0)
print('Mean face computed')

# Subtract the mean face from each individual face image
subtracted_faces = face_images_array - mean_face
print('Faces converted to subtracted')

print(subtracted_faces[0].shape)

# Vectorize the subtracted faces
vectorized_faces = subtracted_faces.reshape(subtracted_faces.shape[0], -1)
print(vectorized_faces[0].shape)
print(vectorized_faces.shape)

# Now 'vectorized_faces' is a 2D array where each row represents a flattened face image
# 'mean_face' is the average face calculated from all the images
