import numpy as np
from imageio import imread
from scipy.io import loadmat
import pandas as pd
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import cv2 as cv
from sklearn.metrics import mean_absolute_error as mae 
import math

def compute_diss_matrix(sim_mat):
    diss_mat = np.zeros_like(sim_mat)
    for i in range(len(sim_mat)):
        for j in range(len(sim_mat)):
            diss_mat[i, j] = math.sqrt(sim_mat[i][i] - 2*sim_mat[i][j] + sim_mat[j][j])

    return diss_mat

def plot_matrix(mat):
    fig, ax = plt.subplots()
    min_val, max_val = 0, 24

    for i in range(24):
        for j in range(24):
            c = mat[i][j]
            ax.text(i+0.5, j+0.5, str(c), va='center', ha='center')

    plt.matshow(mat, cmap=plt.cm.Blues)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(np.arange(max_val))
    ax.set_yticks(np.arange(max_val))
    ax.grid()
    plt.show()


dir_challenge = ""
matrices = loadmat(r'C:\Users\guasc\Documentos\GitHub\ACG_2024_Nuria_Maria\lab3\Lab_3\matrices.mat') # canviar !!!

similarity_matrix = matrices['simScores'][0][0]['similarityM']
consistency_matrix = matrices['simScores'][0][0]['consistencyM']

# COMPARING CONSISTENCY WITH OUR ANNOTATIONS
diff = consistency_matrix - similarity_matrix 
non_infinite_values = diff[np.isfinite(consistency_matrix)]
mae_our_annotations = np.mean(np.mean(np.abs(non_infinite_values)))
print('mae our annotations', mae_our_annotations)

# COMPARING CONSISTENCY WITH RANDOM ANNOTATIONS
mae = []
num_iterations = 15
for i in range(num_iterations):
    random_mat = np.random.randint(0,9,(24,24))
    cons_sim_dif = random_mat - similarity_matrix 
    cons_sim_diff_values = cons_sim_dif[np.isfinite(consistency_matrix)]
    mae.append(np.mean(np.mean(np.abs(cons_sim_diff_values))))

print('mae of random', mae)
exit()

#PLOT O EL QUE SIGUI DE RANDOM MAEs

#DISSIMILARITY MATRIX
diss_mat = compute_diss_matrix(similarity_matrix)

#PLOT SIMILARITY AND DISSIMILARITY MATRIX
plot_matrix(similarity_matrix)
plot_matrix(diss_mat)

#EXTRACT TWO BASES USING MULTIDIMENSIONAL SCALING

#Quina part de la consistency hem d'agafar (amb inf?); primera part -> similarity amb cosistency nostra; segona part -> random sim, random cons? random sim, nostra cons?...
