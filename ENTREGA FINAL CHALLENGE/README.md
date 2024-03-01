# Face Detection and Recognition System

## Objective
The aim of this project is to improve our face detection and recognition system that we delivered for Lab 4, using a custom-built model, specifically a simple Convolutional Neural Network (CNN).

## Folder content
This folder contains the following files:

- `celebrities.py`: A dictionary relating people's names with IDs.
- `CHALl_AGC_FRbasicScript.py`: The main file, containing the code with the improved detection + recognition system.
- `lab1_CHALL_AGC_FDbasicScript.py`: Lab1 face detection file, used for the scrapped images + copied to our function.
- `load_more_images.py`: Code used to scrap more images from different internet search engines to increment our training dataset.
- `process_model_images.py`: Processing for the scrapped images.
- `reduced_100epochs_batch400_resize150_conv64.cpkt`: Weights for our trained model.
- `requirements.txt`: Dependencies for our environment and code to execute.
- `haarcascade_....`: Files used for face detection.
- `grid_search.py`: python script used to find the optimal threshold for assigning -1 label to impostors
- `ID_ESTIMATION_MODEL_FOR_TRAINING.ipynb`: Notebook where the model was defined and trained.

## How to execute code & install requirements
To execute the code, first run the following command in the terminal:

```bash
pip install -r requirements.txt
```
Then, navigate to the lab4 folder and execute the code:

```bash
python Python/CHALL_AGC_FRBasicScript.py
```
