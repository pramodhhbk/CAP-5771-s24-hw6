"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
import pickle
from typing import Tuple,Optional

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> Tuple[Optional[NDArray[np.int32]], Optional[float], Optional[float], Optional[NDArray[np.floating]]]:

    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints.
    Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    def create_shared_neighbor_matrix(data, k, t):
       
        distance_matrix = squareform(pdist(data, 'euclidean'))
        neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
        n = len(data)
        adjacency_matrix = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                shared_neighbors = len(set(neighbors[i]).intersection(neighbors[j]))
                if shared_neighbors >= t:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True
        return adjacency_matrix

    def calculate_sse(data, labels, cluster_centers, cluster_map):
        
        sse = 0.0
        for k, center_index in cluster_map.items():
            if k >= 0:  # Ensure we're only considering valid cluster indices
                cluster_data = data[labels == k]  # Extract data points for the current cluster
                center = cluster_centers[center_index]  # Retrieve the center for the current cluster
                distances = np.linalg.norm(cluster_data - center, axis=1)  # Compute Euclidean distance
                squared_distances = distances**2  # Square the distances
                sse += np.sum(squared_distances)  # Accumulate the SSE

        return sse

    def adjusted_rand_index(labels_true, labels_pred):
        # Find the unique classes and clusters
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)

        # Create the contingency table
        contingency_table = np.zeros((classes.size, clusters.size), dtype=int)
        for class_idx, class_label in enumerate(classes):
            for cluster_idx, cluster_label in enumerate(clusters):
                contingency_table[class_idx, cluster_idx] = np.sum((labels_true == class_label) & (labels_pred == cluster_label))

        # Compute the sum over the rows and columns
        sum_over_rows = np.sum(contingency_table, axis=1)
        sum_over_cols = np.sum(contingency_table, axis=0)

        # Compute the number of combinations of two
        n_combinations = sum([n_ij * (n_ij - 1) / 2 for n_ij in contingency_table.flatten()])
        sum_over_rows_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_rows])
        sum_over_cols_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_cols])

        # Compute terms for the adjusted Rand index
        n = labels_true.size
        total_combinations = n * (n - 1) / 2
        expected_index = sum_over_rows_comb * sum_over_cols_comb / total_combinations
        max_index = (sum_over_rows_comb + sum_over_cols_comb) / 2
        denominator = (max_index - expected_index)

        # Handle the special case when the denominator is 0
        if denominator == 0:
            return 1 if n_combinations == expected_index else 0

        ari = (n_combinations - expected_index) / denominator

        return ari


    def dbscan_custom(matrix, data, minPts):
        
        n = matrix.shape[0]
        pred_labels = -np.ones(n)
        cluster_id = 0
        cluster_centers = []
        cluster_map = {}

        for i in range(n):
            if pred_labels[i] != -1:
                continue
            neighbors = np.where(matrix[i])[0]
            if len(neighbors) < minPts:
                continue  # Skip if not enough points to form a cluster
            seed_set = set(neighbors)
            cluster_points = [data[i]]

            while seed_set:
                current_point = seed_set.pop()
                if pred_labels[current_point] == -2:
                    pred_labels[current_point] = cluster_id
                if pred_labels[current_point] != -1:
                    continue
                pred_labels[current_point] = cluster_id
                current_neighbors = np.where(matrix[current_point])[0]
                if len(current_neighbors) >= minPts:
                    seed_set.update(current_neighbors)

            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
            cluster_map[cluster_id] = len(cluster_centers) - 1
            cluster_id += 1

        return pred_labels, np.array(cluster_centers), cluster_map
    
    k = params_dict['k']
    smin = params_dict['smin']  

    adjacency_matrix = create_shared_neighbor_matrix(data, k,t=2)
    computed_labels, cluster_centers, cluster_map = dbscan_custom(adjacency_matrix, data, smin)
    SSE = calculate_sse(data, computed_labels, cluster_centers, cluster_map)

    ARI = adjusted_rand_index(labels, computed_labels)


 
    return computed_labels, SSE, ARI

def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    answers["jarvis_patrick_function"] = jarvis_patrick

    max_ari = -np.inf  # Smallest possible float
    min_sse = np.inf   # Largest possible float




    # Define the range of parameters for the parameter study
    k_values = np.linspace(3, 8, 6, dtype=int)
    smin_values = np.linspace(4, 10, 7, dtype=int)
    # Load data
    data = np.load('question1_cluster_data.npy')[:5000]
    labels = np.load('question1_cluster_labels.npy')[:5000]
    params_dict = {}
    b_ari = -np.inf
    best_sse = np.inf
    best_ari = {}
    best_params_sse = {}

    # Dictionary to store all results for visualization
    results = []

    # Loop over all combinations of k and smin
    for k in k_values:
        for smin in smin_values:
            params_dict = {'k': k, 'smin': smin}
            computed_labels, SSE, ARI = jarvis_patrick(data[:1000], labels[:1000], params_dict)

            # Store results
            results.append((k, smin, SSE, ARI))

            # Check if this combination gives a higher ARI or lower SSE
            if ARI > b_ari:
                b_ari = ARI
                best_ari = {'k': k, 'smin': smin, 'ARI': ARI}
            if SSE < best_sse:
                best_sse = SSE
                best_params_sse = {'k': k, 'smin': smin, 'SSE': SSE}

   
    best_results = {
        "results": results,
        "best_ari": best_ari,
        "best_params_sse": best_params_sse
    }

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').



    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    final_k = best_ari['k']
    final_smin = best_ari['smin']
    max_ari_data = None
    max_ari_labels = None
    min_sse_data = None
    min_sse_labels = None
    min_sse = np.inf
    max_ari = -np.inf
    for i in range(5):
        params_dict['k'] = final_k
        params_dict['smin'] = final_smin
        index_start = 1000 * i
        index_end = 1000 * (i + 1) 
        computed_labels, SSE, ARI = jarvis_patrick(data[index_start:index_end], labels[index_start:index_end], params_dict)
        groups[i] = {"k": final_k, "smin": final_smin, "ARI": ARI, "SSE": SSE}

        if SSE < min_sse:
            min_sse = SSE
            min_sse_data = data[index_start:index_end]
            min_sse_labels = computed_labels
        
        if ARI > max_ari:
            max_ari = ARI
            max_ari_data = data[index_start:index_end]
            max_ari_labels = computed_labels



    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"]

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    k_values = [item[0] for item in best_results['results']]
    smin_values = [item[1] for item in best_results['results']]
    sses = [item[2] for item in best_results['results']]
    aris = [item[3] for item in best_results['results']]

    plt.scatter(k_values, smin_values, c=aris, cmap='viridis', s=25)
    plt.title('Clusters with Largest ARI')
    plt.xlabel('k')
    plt.ylabel('smin')
    plt.colorbar()
    plt.savefig('Q2_Plot1.png')
    plt.close()

    plt.scatter(k_values, smin_values, c=sses, cmap='viridis', s=25)
    plt.title('Clusters with Smallest SSE')
    plt.xlabel('k')
    plt.ylabel('smin')
    plt.colorbar()
    plt.savefig('Q2_Plot2.png')
    plt.close()


    ARIs = np.array([group["ARI"] for group in groups.values()])
    SSEs = np.array([group["SSE"] for group in groups.values()])
    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter(max_ari_data[:, 0], max_ari_data[:, 1], c=max_ari_labels, cmap='viridis', s=25)
    plt.title('Clusters with Largest ARI')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('Q2_Plot3.png')
    plt.close()
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    
    plot_SSE = plt.scatter(min_sse_data[:, 0], min_sse_data[:, 1], c=min_sse_labels, cmap='viridis', s=25)
    plt.title('Clusters with Smallest SSE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('Q3_Plot4.png')
    plt.close()
    answers["cluster scatterplot with smallest SSE"] = plot_SSE



    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = float(np.mean(ARIs))

    # A single float
    answers["std_ARIs"] = float(np.std(ARIs))

    # A single float
    answers["mean_SSEs"] = float(np.mean(SSEs))

    # A single float
    answers["std_SSEs"] = float(np.std(SSEs))

    return answers




# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
