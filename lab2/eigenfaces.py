import cv2
import os
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

filename = 'vectorized_faces.pkl'
original_height, original_width = 1718, 2444
target_size = 250
ratio = original_width / original_height # we want to mantain the original width/height ratio after resizing
target_width = int(target_size * ratio)
target_height = target_size


if filename not in os.listdir():

    # Directory containing face images
    faces_directory = "dataset_chicago/cfd/CFD Version 3.0/Images/NeutralFaces"
    print('loading images...')
    face_images = []
    for filename in os.listdir(faces_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(faces_directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read images in grayscale
            img_resized = cv2.resize(img, (target_width, target_height))
            face_images.append(img_resized)

            '''
            # If you want to visualize the images after resizing

            plt.imshow(img_resized, cmap='gray', vmin=0, vmax=255)
            plt.title(f"Resized Image")
            plt.axis('off')  # Hide axes
            plt.show()
            '''

    # Convert the list of images to a NumPy array
    face_images_array = np.array(face_images)
    print('Faces converted to array')

    # Calculate the mean face
    mean_face = np.mean(face_images_array, axis=0)
    print('Mean face computed')

    # Subtract the mean face from each individual face image
    subtracted_faces = face_images_array - mean_face

    # Vectorize the subtracted faces
    vectorized_faces = (subtracted_faces.reshape(subtracted_faces.shape[0], -1)).transpose()

    print('Shape of matrix of vectorized images:', vectorized_faces.shape)
    
    cwd = os.getcwd()
    filename = os.path.join(cwd, 'vectorized_faces.pkl')

    pickle.dump(vectorized_faces, open(filename, 'wb'))
    pickle.dump(target_width, open(filename, 'ab'))
    pickle.dump(target_height, open(filename, 'ab'))

    print('Matrix of vectorized images saved.')

else:
    cwd = os.getcwd()
    filename = os.path.join(cwd, 'vectorized_faces.pkl')

    with open(filename, 'rb') as file:
        vectorized_faces = pickle.load(file)
        target_width = pickle.load(file)
        target_height = pickle.load(file)

        print("Shape of matrix:", vectorized_faces.shape)


# 'vectorized_faces' is a 2d array where each column represents a vectorizeed face image
# 'mean_face' is the average face calculated from all the images

#compute pseudo-covariance matrix --> L = Xt*X
L = np.dot(vectorized_faces.T, vectorized_faces)

eigenvalues, eigenvectors = np.linalg.eigh(L)

# Keep only the positive eigenvalues and their corresponding eigenvectors
positive_eigenvalue_indices = eigenvalues > 0
eigenvalues = eigenvalues[positive_eigenvalue_indices]
eigenvectors = eigenvectors[:, positive_eigenvalue_indices]

# Sort eigenvalues and corresponding eigenvectors
sort_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

# Normalize the eigenvectors
normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)


print("Eigenvalues:", eigenvalues.shape)
print("Eigenvectors:", eigenvectors.shape)

'''normalized_first_eigenvector = normalized_eigenvectors[:, 0].reshape(target_height, target_width)

# Display the reshaped normalized eigenvector as an image
plt.imshow(normalized_first_eigenvector, cmap='gray', vmin=np.min(normalized_first_eigenvector), vmax=np.max(normalized_first_eigenvector))
plt.title("First Normalized Eigenface")
plt.axis('off')
plt.show()'''

# Projecting faces onto the eigenspace
projections = np.dot(vectorized_faces, normalized_eigenvectors)
print('Projections:', projections.shape)

# Number of principal components to retain (you can adjust this)
num_components = 10

# Select the top 'num_components' eigenvectors and projections
top_eigenvectors = normalized_eigenvectors[:, :num_components]
top_projections = projections[:, :num_components]

# Reconstruct the faces using the top eigenvectors and projections
reconstructed_faces = np.dot(top_projections, top_eigenvectors.T)
print('Reconstructed faces:', reconstructed_faces.shape)

# Reshape the reconstructed faces to their original size
reshaped_faces = reconstructed_faces.T.reshape((num_components, target_height, target_width))

# Add the mean face back to obtain the final reconstructed faces
final_reconstructed_faces = reshaped_faces + mean_face.flatten()

# Display the original and reconstructed faces for visualization
plt.figure(figsize=(12, 6))
for i in range(num_components):
    plt.subplot(2, num_components, i + 1)
    plt.imshow(face_images_array[i], cmap='gray', vmin=0, vmax=255)
    plt.title(f"Original {i + 1}")
    plt.axis('off')

    plt.subplot(2, num_components, num_components + i + 1)
    plt.imshow(final_reconstructed_faces[i], cmap='gray', vmin=0, vmax=255)
    plt.title(f"Reconstructed {i + 1}")
    plt.axis('off')

plt.show()

#PLOTTING THE EIGENVALUES OF OUR PCA vs RANDOM -> NO FUNCIONA, PETA LA TERMINAL
num_noise_images = 100
image_height, image_width = target_height, target_width  # Assuming the size of your images

# Generate random noise images
random_noise_images = np.random.randint(0, 256, size=(num_noise_images, image_height, image_width), dtype=np.uint8)

# Vectorize the random noise images
vectorized_noise_images = random_noise_images.reshape((num_noise_images, -1))

# Perform PCA on vectorized noise images
cov_matrix_noise = np.cov(vectorized_noise_images.T)
eigenvalues_noise, _ = np.linalg.eigh(cov_matrix_noise)

# Sort eigenvalues in descending order
eigenvalues_noise = eigenvalues_noise[::-1]

plt.figure(figsize=(12, 6))

# Bar plot for facial images eigenvalues
plt.subplot(1, 2, 1)
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='blue')
plt.title('Eigenvalues - Facial Images')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude')
plt.yscale('log')  # Use log scale for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Bar plot for random noise images eigenvalues
plt.subplot(1, 2, 2)
plt.bar(range(1, len(eigenvalues_noise) + 1), eigenvalues_noise, color='red')
plt.title('Eigenvalues - Random Noise Images')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude')
plt.yscale('log')  # Use log scale for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

