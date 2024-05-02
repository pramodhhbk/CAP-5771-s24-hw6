"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.linalg import eigh
from scipy.cluster.vq import kmeans2
import matplotlib.backends.backend_pdf as pdf
from typing import Tuple,Optional

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################




def plot_clustering(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def proximity_measure(x, y, sigma):

    dist_squared = np.sum((x - y) ** 2)
    return np.exp(-dist_squared / (2 * sigma ** 2))

def adjusted_rand_index(true_labels, pred_labels):
    """
    Compute the adjusted Rand index.

    Arguments:
    - true_labels: true labels of the data
    - pred_labels: predicted labels by the clustering algorithm

    Return value:
    - Adjusted Rand index
    """
    contingency_matrix = np.zeros((np.max(true_labels) + 1, np.max(pred_labels) + 1), dtype=np.int64)
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i], pred_labels[i]] += 1

    a = np.sum(contingency_matrix, axis=1)
    b = np.sum(contingency_matrix, axis=0)
    n = np.sum(contingency_matrix)
    ab_sum = np.sum(a * (a - 1)) / 2
    cd_sum = np.sum(b * (b - 1)) / 2
    a_plus_b_choose_2 = np.sum(a * (a - 1)) / 2
    c_plus_d_choose_2 = np.sum(b * (b - 1)) / 2

    ad_bc = np.sum(contingency_matrix * (contingency_matrix - 1)) / 2

    expected_index = a_plus_b_choose_2 * c_plus_d_choose_2 / n / (n - 1) + ad_bc ** 2 / n / (n - 1)
    max_index = (a_plus_b_choose_2 + c_plus_d_choose_2) / 2
    return (ad_bc - expected_index) / (max_index - expected_index)

def compute_SSE(data, labels):
 
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> Tuple[Optional[NDArray[np.int32]], Optional[float], Optional[float], Optional[NDArray[np.floating]]]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """

    sigma = params_dict['sigma']
    k = params_dict['k']

    # Construct the similarity matrix
    n_samples = data.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            similarity_matrix[i, j] = proximity_measure(data[i], data[j], sigma)

    # Construct the Laplacian matrix
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    # Perform k-means clustering on the eigenvectors with k-means++ initialization
    _, computed_labels = kmeans2(eigenvectors[:, 1:k], k, minit='++')

    # Compute SSE
    SSE = compute_SSE(data, computed_labels)

    # Compute adjusted Rand index
    ARI = adjusted_rand_index(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues

def spectral_hyperparameter_study(data, labels):
    """
    Perform hyperparameter study for spectral clustering on the given data.

    Arguments:
    - data: input data array of shape (n_samples, n_features)
    - labels: true labels of the data

    Return values:
    - sigmas: Array of sigma values
    - ari_scores: Array of ARI scores for each sigma value
    - sse_scores: Array of SSE scores for each sigma value
    """
    ari_scores = []
    sse_scores = []

    sigmas = np.logspace(-1, 1, num=10)  
    k = 5  

    for sigma in sigmas:
        _, sse, ari, _ = spectral(data, labels, {'sigma': sigma, 'k': k})
        sse_scores.append(sse)
        ari_scores.append(ari)

    return sigmas, np.array(ari_scores), np.array(sse_scores)

def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    
    cluster_data = np.load('question1_cluster_data.npy')
    cluster_labels = np.load('question1_cluster_labels.npy')
    
    data_subset = cluster_data[:1000]
    labels_subset = cluster_labels[:1000]
    
    sigmas, ari_scores, sse_scores = spectral_hyperparameter_study(data_subset, labels_subset)
    
    
    
    # Plot ARI scores against sigma values
    plt.figure(figsize=(8, 6))
    plt.plot(sigmas, ari_scores, color='b')
    plt.title('ARI vs Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('ARI')
    plt.grid(True)
    plt.xscale('log') 
    plt.savefig('Plot_1.png')
    plt.close()
    
    # Plot SSE scores against sigma values
    plt.figure(figsize=(8, 6))
    plt.plot(sigmas, sse_scores, color='r')
    plt.title('SSE vs Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.xscale('log') 
    plt.savefig('Plot_2.png')
    plt.close()
    
    
    # After hyperparameter study
    best_sigma = 0.1
    best_k = 5
    
    plots_values={}
    # Apply best hyperparameters on five slices of data
    for i in [0,1,2,3,4]:
        data_slice = cluster_data[i * 1000: (i + 1) * 1000]
        labels_slice = cluster_labels[i * 1000: (i + 1) * 1000]
        computed_labels, sse, ari, eig_values = spectral(data_slice, labels_slice, {'sigma': best_sigma, 'k': best_k})
        groups[i] = {"sigma": best_sigma, "ARI": ari, "SSE": sse}
        plots_values[i] = {"computed_labels": computed_labels, "ARI": ari, "SSE": sse,"eig_values":eig_values} 
        
    highest_ari = -1
    best_dataset_index = None
    for i, group_info in plots_values.items():
        if group_info['ARI'] > highest_ari:
            highest_ari = group_info['ARI']
            best_dataset_index = i
            
    
    # Plot the clusters for the dataset with the highest ARI
    plt.figure(figsize=(8, 6))
    plot_ARI = plt.scatter(cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 0], 
                cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 1], 
                c=plots_values[best_dataset_index]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for highest ARI for:{best_k}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.savefig('Plot_3.png')
    plt.close()
    
    # Find the dataset with the lowest SSE
    lowest_sse = float('inf')
    best_dataset_index_sse = None
    for i, group_info in plots_values.items():
        if group_info['SSE'] < lowest_sse:
            lowest_sse = group_info['SSE']
            best_dataset_index_sse = i
    
    # Plot the clusters for the dataset with the lowest SSE
    plt.figure(figsize=(8, 6))
    plot_SSE = plt.scatter(cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 0], 
                cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 1], 
                c=plots_values[best_dataset_index_sse]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for lowest SSE with k :{best_k}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.savefig('Plot_4.png') 
    plt.close()
    
    
    # Plot of the eigenvalues (smallest to largest) as a line plot for all datasets
    plt.figure(figsize=(8, 6))
    
    for i, group_info in plots_values.items():
        plot_eig = plt.plot(np.sort(group_info["eig_values"]), label=f'Dataset {i+1}')
    
    plt.title('Eigenvalues Plot')
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plot_5.png') 
    plt.close()
    
 
    

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"] #{}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    #plot_ARI = plt.scatter([1,2,3], [4,5,6])
    #plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    #plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    
    ari_values = [group_info["ARI"] for group_info in groups.values()]
    mean_ari = np.mean(ari_values)
    std_dev_ari = np.std(ari_values)

    # A single float
    answers["mean_ARIs"] = mean_ari

    # A single float
    answers["std_ARIs"] = std_dev_ari
    
    sse_values = [group_info["SSE"] for group_info in groups.values()]
    mean_sse = np.mean(sse_values)
    std_dev_sse = np.std(sse_values)

    # A single float
    answers["mean_SSEs"] = mean_sse

    # A single float
    answers["std_SSEs"] = std_dev_sse

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)