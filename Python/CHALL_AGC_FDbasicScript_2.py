import numpy as np
from imageio import imread
from scipy.io import loadmat
import pandas as pd
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import cv2 as cv


def CHALL_AGC_ComputeDetScores(DetectionSTR, AGC_Challenge1_STR, show_figures):
    #  Compute face detection score
    #
    #   INPUTS
    #     - DetectionSTR: A structure with the results of the automatic detection
    #     algorithm, with one element per input image containing field
    #     'det_faces'. This field contains as many 4-column rows as faces
    #     returned by the detector, each specifying a bounding box coordinates
    #     as [x1,y1,x2,y2], with x1 < x2 and y1 < y2.

    #     COORDINATES OF THE FACE DETECTION SQAURES
    #
    #     - AGC_Challenge1_STR: The ground truth structure (e.g.AGC_Challenge1_TRAINING or AGC_Challenge1_TEST).
    #
    #     - show_figures: A flag to enable detailed displaying of the results for
    #     each input image. If set to zero it just conputes the scores, with no
    #     additional displaying.
    #
    #   OUTPUT
    #     - FD_score:     The final detection score obtained by the detector
    #     - scoresSTR:    Structure with additional detection information
    #   --------------------------------------------------------------------
    #   AGC Challenge
    #   Universitat Pompeu Fabra
    #
    feature_list = ['F1', 'Fmatrix']
    values = np.zeros((len(AGC_Challenge1_STR), 2), dtype='float')
    scoresSTR = pd.DataFrame(values, index=np.arange(len(AGC_Challenge1_STR)), columns=feature_list, dtype='object')
    for i in range(0, len(AGC_Challenge1_STR)):
        if show_figures:
            A = imread(AGC_Challenge1_STR['imageName'][i])
            fig, ax = plt.subplots()
            ax.imshow(A)
            for k1 in range(0, len(AGC_Challenge1_STR['faceBox'][i])):
                if len(AGC_Challenge1_STR['faceBox'][i]) != 0:
                    bbox = np.array(AGC_Challenge1_STR['faceBox'][i][k1], dtype=int)
                    fb = Rectangle((bbox[0], bbox[3]), bbox[2] - bbox[0], bbox[1] - bbox[3], linewidth=4, edgecolor='b',
                                   facecolor='none')
                    ax.add_patch(fb)
            for k2 in range(0, len(DetectionSTR[i])):
                if len(DetectionSTR[i]) != 0:
                    bbox = np.array(DetectionSTR[i][k2], dtype=int)
                    fb = Rectangle((bbox[0], bbox[3]), bbox[2] - bbox[0], bbox[1] - bbox[3], linewidth=4, edgecolor='g',
                                   facecolor='none')
                    ax.add_patch(fb)
        n_actualFaces = len(AGC_Challenge1_STR['faceBox'][i])
        n_detectedFaces = len(DetectionSTR[i])
        if not n_actualFaces:
            if n_detectedFaces:
                scoresSTR['F1'][i] = np.zeros(n_detectedFaces)
            else:
                scoresSTR['F1'][i] = np.array([1], dtype=float)
        else:
            if not n_detectedFaces:
                scoresSTR['F1'][i] = np.zeros(n_actualFaces)
            else:
                scoresSTR['Fmatrix'][i] = np.zeros((n_actualFaces, n_detectedFaces))
                for k1 in range(0, n_actualFaces):
                    f = np.array(AGC_Challenge1_STR['faceBox'][i][k1], dtype=int)
                    for k2 in range(0, n_detectedFaces):
                        g = np.array(DetectionSTR[i][k2], dtype=int)
                        # Intersection box
                        x1 = max(f[0], g[0])
                        y1 = max(f[1], g[1])
                        x2 = min(f[2], g[2])
                        y2 = min(f[3], g[3])
                        # Areas
                        int_Area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                        total_Area = (f[2] - f[0]) * (f[3] - f[1]) + (g[2] - g[0]) * (g[3] - g[1]) - int_Area
                        if n_detectedFaces == 1 and n_actualFaces == 1:
                            scoresSTR['Fmatrix'][i] = int_Area / total_Area
                        else:
                            scoresSTR['Fmatrix'][i][k1, k2] = int_Area / total_Area
                scoresSTR['F1'][i] = np.zeros((max(n_detectedFaces, n_actualFaces)))
                for k3 in range(0, min(n_actualFaces, n_detectedFaces)):
                    max_F = np.max(scoresSTR['Fmatrix'][i])
                    if n_detectedFaces == 1 and n_actualFaces == 1:
                        scoresSTR['F1'][i] = np.array([max_F], dtype=float)
                        scoresSTR['Fmatrix'][i] = 0
                        scoresSTR['Fmatrix'][i] = 0
                    else:
                        max_ind = np.unravel_index(np.argmax(scoresSTR['Fmatrix'][i], axis=None), scoresSTR['Fmatrix'][i].shape)
                        scoresSTR['F1'][i][max_ind[1]] = max_F
                        scoresSTR['Fmatrix'][i][max_ind[0], :] = 0
                        scoresSTR['Fmatrix'][i][:, max_ind[1]] = 0
        if show_figures:
            try:
                plt.title("%.2f" % scoresSTR['F1'][i])
            except:
                plt.title('%.2f, %.2f' % (scoresSTR['F1'][i][0], scoresSTR['F1'][i][1]))
                
            plt.show()
            plt.clf()
            plt.close()

    FD_score = np.mean(np.hstack(np.array(scoresSTR['F1'][:])))
    return FD_score


def MyFaceDetectionFunction(A, name):
    # Function to implement
    if not len(A.shape) == 2:
        grayscale = cv.cvtColor(A, cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    eyeglasses_cascade = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    nose_cascade = cv.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')
    smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')
    profile_cascade = cv.CascadeClassifier('haarcascade_profileface.xml')
    cat_cascade = cv.CascadeClassifier('haarcascade_frontalcatface.xml')
    
    
    detected_faces = haar_cascade.detectMultiScale(grayscale, scaleFactor=1.2, minNeighbors=3) 
    # scaleFactor: how much the image is reduced at each step. 1.05 --> 5%

    valid_faces = [] # use other feature detections to filter out false positives

    for (x, y, w, h) in detected_faces:
        face_roi = grayscale[y:y+h, x:x+w]
        detected_eyes = eye_cascade.detectMultiScale(face_roi)
        detected_eyeglasses = eyeglasses_cascade.detectMultiScale(face_roi)
        detected_nose = nose_cascade.detectMultiScale(face_roi)
        detected_mouth = mouth_cascade.detectMultiScale(face_roi)
        detected_smile = smile_cascade.detectMultiScale(face_roi)
        detected_profile = profile_cascade.detectMultiScale(face_roi)
        detected_cat = cat_cascade.detectMultiScale(face_roi)
    
        all_detected =  len(detected_eyes) + len(detected_nose) + len(detected_mouth) + len(detected_eyeglasses)# - len(detected_cat) + len(detected_profile)
        # len(detected_faces) +

        #print(name, ':', all_detected)

        if all_detected >= 4: # amb 5 --> 80.64
            #valid_faces.append([int(x), int(y), int(x + w), int(y + h)])
            valid_faces.append([int(x), int(y), int(x + w), int(y + h), all_detected])

        #if len(detected_eyes) > 0 or len(detected_eyeglasses) > 0:
            #valid_faces.append([int(x), int(y), int(x + w), int(y + h)])
    
   # Sort faces based on size (width * height)
    #valid_faces = sorted(valid_faces, key=lambda x: x[2] * x[3], reverse=True)

    # Keep only the two largest faces
    #valid_faces = valid_faces[:2]

    valid_faces = sorted(valid_faces, key=lambda x: x[4], reverse=True)

    # Keep only the two largest faces without the all_detected score
    valid_faces = valid_faces[:2]
    valid_faces = [[x[0], x[1], x[2], x[3]] for x in valid_faces]  # Remove the all_detected score
    
    return valid_faces


# Basic script for Face Detection Challenge
# --------------------------------------------------------------------
# AGC Challenge
# Universitat Pompeu Fabra

# Load challenge Training data
dir_challenge = ""
AGC_Challenge1_TRAINING = loadmat(dir_challenge + "AGC_Challenge1_Training.mat")

AGC_Challenge1_TRAINING = np.squeeze(AGC_Challenge1_TRAINING['AGC_Challenge1_TRAINING'])
AGC_Challenge1_TRAINING = [[row.flat[0] if row.size == 1 else row for row in line] for line in AGC_Challenge1_TRAINING]
columns = ['id', 'imageName', 'faceBox']
AGC_Challenge1_TRAINING = pd.DataFrame(AGC_Challenge1_TRAINING, columns=columns)

# Provide the path to the input images, for example
# 'C:/AGC_Challenge/images/'
imgPath = "TRAINING/"
AGC_Challenge1_TRAINING['imageName'] = imgPath + AGC_Challenge1_TRAINING['imageName'].astype(str)
# Initialize results structure
DetectionSTR = []

# Initialize timer accumulator
total_time = 0
for idx, im in enumerate(AGC_Challenge1_TRAINING['imageName']):
    A = imread(im)
    try:
        ti = time.time()
        # Timer on
        ###############################################################
        # Your face detection function goes here. It must accept a single
        # input parameter (the input image A) and it must return one or
        # more bounding boxes corresponding to the facial images found
        # in image A, specificed as [x1 y1 x2 y2]
        # Each bounding box that is detected will be indicated in a
        # separate row in det_faces

        det_faces = MyFaceDetectionFunction(A, im)

        tt = time.time() - ti
        total_time = total_time + tt
    except:
        # If the face detection function fails, it will be assumed that no
        # face was detected for this input image
        det_faces = []

    DetectionSTR.append(det_faces)

FD_score = CHALL_AGC_ComputeDetScores(DetectionSTR, AGC_Challenge1_TRAINING, show_figures=True)
_, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print('F1-score: %.2f, Total time: %2d m %.2f s' % (100 * FD_score, int(minutes), seconds))
