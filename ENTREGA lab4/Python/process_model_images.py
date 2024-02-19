
import cv2 as cv
from lab1_CHALL_AGC_FDbasicScript import my_face_detection
import scipy.io
import pandas as pd
from celebrities import celebrities
import os

#AFEGIR AL MAT LES FOTOS NOVES -> NOM + ID -> bounding box []
#separar .mat en test i train
#agafr train -> preprocess
# comprovar si tenen bounding box -> CROP 
# la resta (les noves) -> PASSAR PER LA NOSTRA FUNCIO DE DETECTAR -> CROP
# si no en tenen i l'id -1 -> RES
#FER RESIZE DE TOTES -> per tenir mateixa mida
#GUARDAR AMB ZIP I PASSAR AL MODEL

#  1. afegir dades del mat file + scrapped a un df

# Load the existing .mat file into a DataFrame
data = scipy.io.loadmat('AGC_Challenge3_Training.mat')
df = pd.DataFrame(data)

# Initialize a list to store dictionaries representing new rows
new_rows = []

# Iterate over each celebrity and their corresponding images
for celebrity, celebrity_id in celebrities.items(): # diccionari
    folder_path = f'photos/{celebrity}'
    if os.path.exists(folder_path):
        # Get a list of image files in the folder
        image_files = os.listdir(folder_path)
        
        # Add a new row for each image
        for image_file in image_files:
            new_row = {
                'image_name': image_file,
                'bounding_box': [],  # Assuming an empty bounding box is represented as an empty list
                'person_id': celebrity_id
            }
            new_rows.append(new_row)

# Append the new rows to the DataFrame
df = df.append(new_rows, ignore_index=True)


'''
if ...
    crop
else if ...
    NOSTRA funcio

resize
'''

