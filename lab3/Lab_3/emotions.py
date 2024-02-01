import numpy as np
from imageio import imread
from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error as mae 
import math
from sklearn.manifold import MDS

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

def plot_comparison(random, our):
    num_iterations = 20
    random_mae_values = np.array(random)

    # Plotting the bars
    plt.bar(range(num_iterations), random_mae_values, label='MAE of Random Annotations')

    # Drawing a horizontal line for mae_our_annotations
    plt.axhline(y=our, color='r', linestyle='-', label='MAE of Our Annotations')

    # Adding labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Comparison of MAE between Our Annotations and Random Annotations')
    plt.legend()

    # Show the plot
    plt.show()

def plot_bases(bases):
    plt.scatter(bases[:, 0], bases[:, 1], label='Data Points')

    # Add labels and title
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Scatter Plot of Data Points in Reduced Space')

    # Show legend if needed
    plt.legend()

    # Show the plot
    plt.show()


dir_challenge = ""
matrices = loadmat(r'/Users/nuriacodina/Desktop/UPF/QUART/2N_TRIM/FGA/ACG_2024_Nuria_Maria/lab3/Lab_3/matrices.mat')
#matrices = loadmat(r'C:\Users\guasc\Documentos\GitHub\ACG_2024_Nuria_Maria\lab3\Lab_3\matrices.mat') # canviar !!!

similarity_matrix = matrices['simScores'][0][0]['similarityM']
consistency_matrix = matrices['simScores'][0][0]['consistencyM']

# COMPARING CONSISTENCY WITH OUR ANNOTATIONS
diff = consistency_matrix - similarity_matrix 
non_infinite_values = diff[np.isfinite(consistency_matrix)]
mae_our_annotations = np.mean(np.mean(np.abs(non_infinite_values)))
#print('mae our annotations', mae_our_annotations)

# COMPARING CONSISTENCY WITH RANDOM ANNOTATIONS
mae = []
num_iterations = 20
for i in range(num_iterations):
    random_mat = np.random.randint(0,9,(24,24))
    cons_sim_dif = random_mat - similarity_matrix 
    cons_sim_diff_values = cons_sim_dif[np.isfinite(consistency_matrix)]
    mae.append(np.mean(np.mean(np.abs(cons_sim_diff_values))))

#print('mae of random', mae)

plot_comparison(mae, mae_our_annotations)

#PLOT O EL QUE SIGUI DE RANDOM MAEs

#DISSIMILARITY MATRIX
diss_mat = compute_diss_matrix(similarity_matrix)

#PLOT SIMILARITY AND DISSIMILARITY MATRIX
plot_matrix(similarity_matrix)
plot_matrix(diss_mat)

#EXTRACT TWO BASES USING MULTIDIMENSIONAL SCALING
A = -0.5 * (diss_mat ** 2) #Compute A
#Compute necessary variables to calculate B
n = len(diss_mat)
I_n = np.identity(n)
J_n = np.ones((n, n))
H = I_n - (1/n) * J_n
B = np.dot(np.dot(H, A), H) #Compute the doubly-centered symmetric matrix B

#Compute eigenvalues & eigenvectors of B
eigenvalues, eigenvectors = np.linalg.eig(B)
# Sort eigenvalues and their corresponding eigenvectors
sort_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

first_two_bases = eigenvectors[:, :2]
plot_bases(first_two_bases)

#COMPROVACIÓ AMB FUNCIO JA CREADA
mds = MDS(n_components=2, dissimilarity='precomputed')

# Fit and transform the dissimilarity matrix
mds_result = mds.fit_transform(diss_mat)
plot_bases(mds_result)
