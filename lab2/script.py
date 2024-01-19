import os
import shutil

# SCRIPT TO EXCTRACT LANDMARKED IMAGES FROM FOLDERS

# Source directory containing subfolders with images
source_directory = "/Users/nuriacodina/Desktop/UPF/QUART/2N_TRIM/FGA/ACG_2024_Nuria_Maria/lab2/dataset_chicago/cfd/CFD Version 3.0/Images/CFD"

# Destination directory where you want to gather images with neutral faces
destination_directory = "/Users/nuriacodina/Desktop/UPF/QUART/2N_TRIM/FGA/ACG_2024_Nuria_Maria/lab2/dataset_chicago/cfd/CFD Version 3.0/Images/NeutralFaces"

# Ensure the destination directory exists, if not, create it
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Traverse the source directory
for root, dirs, files in os.walk(source_directory):
    for file in files:
        # Check if the file is an image and has a neutral face
        if file.lower().endswith('-n.jpg'):
            # Build the full path of the source file
            source_file_path = os.path.join(root, file)
            
            # Build the full path of the destination file
            destination_file_path = os.path.join(destination_directory, file)
            
            # Move or copy the file to the destination directory
            shutil.copy(source_file_path, destination_file_path)
            # If you prefer to move instead of copy, uncomment the line below
            # shutil.move(source_file_path, destination_file_path)

print("Neutral face images have been gathered in the destination directory:", destination_directory)
