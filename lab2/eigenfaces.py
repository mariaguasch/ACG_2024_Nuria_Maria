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
            img_resized = cv2.resize(img, (target_width, target_height)) # img_resized shape (250, 355)
            face_images.append(img_resized)
            if len(face_images) == 1 : print(img_resized.shape)

            '''
            # If you want to visualize the images after resizing

            plt.imshow(img_resized, cmap='gray', vmin=0, vmax=255)
            plt.title(f"Resized Image")
            plt.axis('off')  # Hide axes
            plt.show()
            '''

    # Convert the list of images to a NumPy array
    face_images_array = np.array(face_images) # shape (597, 250, 355)
    print('Shape face_images_array:', face_images_array.shape)

    # Calculate the mean face
    mean_face = np.mean(face_images_array, axis=0) # shape (250, 355)
    print('Mean face shape', mean_face.shape)

    # Subtract the mean face from each individual face image
    subtracted_faces = face_images_array - mean_face
    print('Subtracted faces shape:', subtracted_faces.shape)

    # Vectorize the subtracted faces
    vectorized_faces = (subtracted_faces.reshape(subtracted_faces.shape[0], -1)).transpose()
    print('Shape matrix of vectorized images:', vectorized_faces.shape) # ok
    
    cwd = os.getcwd()
    filename = os.path.join(cwd, 'vectorized_faces.pkl')

    pickle.dump(vectorized_faces, open(filename, 'wb'))
    pickle.dump(target_width, open(filename, 'ab'))
    pickle.dump(target_height, open(filename, 'ab'))
    pickle.dump(face_images, open(filename, 'ab'))
    print('Data saved.')

else:
    cwd = os.getcwd()
    filename = os.path.join(cwd, 'vectorized_faces.pkl')

    with open(filename, 'rb') as file:
        vectorized_faces = pickle.load(file)
        target_width = pickle.load(file)
        target_height = pickle.load(file)
        face_images = pickle.load(file)

        print("Shape of matrix:", vectorized_faces.shape) # ok


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
normalized_eigenvalues = eigenvalues / eigenvalues[0]

print("Eigenvalues:", eigenvalues.shape)
print("Eigenvectors:", eigenvectors.shape)

components = 50

##### VISUALIZATION OF EIGENVALUES #######
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot for eigenvalues
axs[0].bar(range(1, components + 1), normalized_eigenvalues[:components])
axs[0].set_title("Eigenvalues (Bar Plot)")
axs[0].set_xlabel("Eigenvalue Index")
axs[0].set_ylabel("Eigenvalue Magnitude")

# Plot the eigenvalues as a line plot
axs[1].plot(range(1, components + 1), normalized_eigenvalues[:components], marker='o', linestyle='-')
axs[1].set_title("Eigenvalues (Line Plot)")
axs[1].set_xlabel("Eigenvalue Index")
axs[1].set_ylabel("Eigenvalue Magnitude")

fig.suptitle("Eigenvalues of original images")
plt.tight_layout()
plt.show()
#########################################

# Projecting faces onto the eigenspace
projections = np.dot(vectorized_faces, normalized_eigenvectors)
print('Projections:', projections.shape)

'''for i in range(len(normalized_eigenvectors)):
    face = vectorized_faces[:, 0].reshape(target_height, target_width)
    plt.imshow(face, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()'''


###### project 1 face into 1st eigenvector ######
projected_1 = np.dot(vectorized_faces[:,0], normalized_eigenvectors[:, 0])
#################################################

# Number of principal components to retain (you can adjust this)
num_components = 10

# Select the top 'num_components' eigenvectors and projections
top_eigenvectors = normalized_eigenvectors[:, :num_components]
top_projections = projections[:, :num_components]
top_eigenvalues = eigenvalues[:num_components]

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




#PLOTTING THE EIGENVALUES OF OUR PCA vs RANDOM
num_noise_images = 100
image_height, image_width = target_height, target_width  # Assuming the size of your images

# Generate random noise images
noise_images = []
for face in face_images:
    height, width = face.shape
    vector_face = face.flatten()
    np.random.shuffle(vector_face)
    shuffled_image = vector_face.reshape((height, width))
    noise_images.append(shuffled_image)
    
    if len(noise_images) == 1: # display 1 example of face image to randomized noisy image
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].imshow(face, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(shuffled_image, cmap='gray', vmin=0, vmax=255)
        axs[1].set_title('Noise-like Image')
        axs[1].axis('off')
        plt.show()

print('shape of noisy images:', noise_images[0].shape)
random_noise_images = np.array(noise_images)
vectorized_noise_images = random_noise_images.reshape((random_noise_images.shape[0], -1)).transpose()

print('\nComputing eigendecomposition for randomized images...')
r_L = np.dot(vectorized_noise_images.T, vectorized_noise_images)
r_eigenvalues, r_eigenvectors = np.linalg.eigh(r_L)

# Keep only the positive eigenvalues and their corresponding eigenvectors
positive_eigenvalue_indices = r_eigenvalues > 0
print('Number of positive eigenvalues for noisy images:', len(positive_eigenvalue_indices))

r_eigenvalues = r_eigenvalues[positive_eigenvalue_indices]
r_eigenvectors = r_eigenvectors[:, positive_eigenvalue_indices]

# Sort eigenvalues and corresponding eigenvectors
sort_indices = np.argsort(r_eigenvalues)[::-1]
r_eigenvalues = r_eigenvalues[sort_indices]
r_eigenvectors = r_eigenvectors[:, sort_indices]

r_normalized_eigenvectors = r_eigenvectors / np.linalg.norm(r_eigenvectors, axis=0)
r_normalized_eigenvalues = r_eigenvalues / r_eigenvalues[0]

print('Random eigenvalues', r_eigenvalues.shape)
print('Random eigenvectors', r_eigenvectors.shape)

###### Plots for eigenvalue visualization #####
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].bar(range(1, components + 1), r_normalized_eigenvalues[:components])
axs[0].set_title("Eigenvalues (Bar Plot)")
axs[0].set_xlabel("Eigenvalue Index")
axs[0].set_ylabel("Eigenvalue Magnitude")

axs[1].plot(range(1, components + 1), r_normalized_eigenvalues[:components], marker='o', linestyle='-')
axs[1].set_title("Eigenvalues (Line Plot)")
axs[1].set_xlabel("Eigenvalue Index")
axs[1].set_ylabel("Eigenvalue Magnitude")
fig.suptitle("Eigenvalues of randomized images")
plt.tight_layout()
plt.show()
################################################

# Comparison of line plots --> check statistically significant components
plt.plot(range(1, components + 1), r_normalized_eigenvalues[:components], marker='o', linestyle='-', label='Randomized Images')
plt.plot(range(1, components + 1), normalized_eigenvalues[:components], marker='o', linestyle='-', label='Original Data')

plt.title("Eigenvalues Comparison")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Eigenvalue Magnitude")

plt.legend()
plt.show()

# DUBTE: NO SÉ SI PER VEURE LES STATISTICALLY SIGNIFICANT COMPONENTS HEM DE UTILITZAR ELS EIGENVALUES ORIGINALS O NORMALIZATS
# Crec q si els normalizats perq si no totes les components són significants

'''
We can see that the magnitude of the eigenvalues from the original data is larger than the
magnitude of the randomized ones. Then, we can say that they are statistically significant.
'''