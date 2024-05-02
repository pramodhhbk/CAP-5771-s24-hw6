import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple,Optional
from typing import Union
# from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------


def compute_SSE(data, labels):
    """
    Calculate the sum of squared errors (SSE) for a clustering.

    Parameters:
    - data: numpy array of shape (n, 2) containing the data points
    - labels: numpy array of shape (n,) containing the cluster assignments

    Returns:
    - sse: the sum of squared errors
    """
    # ADD STUDENT CODE
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse


def compute_ARI(confusion_matrix: NDArray[np.int32]):
    """
    Compute the Adjusted Rand Index (ARI) metric for evaluating the performance of a clustering algorithm.
    (I AM NOT CONVINCED THE RESULTS ARE CORRECT)

    Parameters:
    confusion_matrix (numpy.ndarray): The confusion matrix representing the clustering results.

    Returns:
    float: The computed ARI value.

    The ARI is a measure of the similarity between two data clusterings. It takes into account both the similarity of the
    clusters themselves and the similarity of the cluster assignments. A higher ARI value indicates a better clustering
    performance.

    The ARI is computed using the formula:
    ARI = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    where:
    - tp: True positives (number of pairs of samples that are in the same cluster in both the true and predicted labels)
    - tn: True negatives (number of pairs of samples that are not in the same cluster in both the true and predicted labels)
    - fp: False positives (number of pairs of samples that are in the same cluster in the true labels but not in the predicted labels)
    - fn: False negatives (number of pairs of samples that are not in the same cluster in the true labels but in the predicted labels)
    """
    tp = confusion_matrix[0, 0]
    tn = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    ari = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ari


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """
    Compute the adjusted Rand index.

    Parameters:
    - labels_true: The true labels of the data points.
    - labels_pred: The predicted labels of the data points.

    Returns:
    - ari: The adjusted Rand index value.

    The adjusted Rand index is a measure of the similarity between two data clusterings.
    It takes into account both the similarity of the clusters themselves and the similarity
    of the data points within each cluster. The adjusted Rand index ranges from -1 to 1,
    where a value of 1 indicates perfect agreement between the two clusterings, 0 indicates
    random agreement, and -1 indicates complete disagreement.
    """
    # Create contingency table
    contingency_table = np.histogram2d(
        labels_true,
        labels_pred,
        bins=(np.unique(labels_true).size, np.unique(labels_pred).size),
    )[0]

    # Sum over rows and columns
    sum_combinations_rows = np.sum(
        [np.sum(nj) * (np.sum(nj) - 1) / 2 for nj in contingency_table]
    )
    sum_combinations_cols = np.sum(
        [np.sum(ni) * (np.sum(ni) - 1) / 2 for ni in contingency_table.T]
    )

    # Sum of combinations for all elements
    N = np.sum(contingency_table)
    sum_combinations_total = N * (N - 1) / 2

    # Calculate ARI
    ari = (
        np.sum([np.sum(n_ij) * (np.sum(n_ij) - 1) / 2 for n_ij in contingency_table])
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    ) / (
        (sum_combinations_rows + sum_combinations_cols) / 2
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    )

    return ari


# Example usage:
true_labels = np.array([0, 0, 1, 1, 0, 1])
pred_labels = np.array([1, 1, 0, 0, 1, 0])
print(adjusted_rand_index(true_labels, pred_labels))

# ----------------------------------------------------------------------


def multivariate_pdf(data, mean, cov):
    """Calculate the probability density function of a multivariate normal distribution."""
    mean = np.asarray(mean).flatten()  # Ensure mean is a 1D array for operations
    cov = np.asarray(cov)
    k = mean.size

    assert data.shape[1] == k, "Data and mean dimensions do not match"
    assert (
        mean.size == cov.shape[0] == cov.shape[1]
    ), "Mean and covariance dimensions do not match"

    # Calculate the inverse and determinant of the covariance matrix
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** k * det_cov))

    # Difference between data points and mean
    diff = data - mean
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)  # Reshape diff to 2D if it's 1D

    # Exponent term in the multivariate normal PDF
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

    # Return the PDF value
    return norm_const * np.exp(exponent)


# ----------------------------------------------------------------------


def extract_samples(
    data: NDArray[np.floating], labels: NDArray[np.int32], num_samples: int
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """
    Extract random samples from data and labels.

    Arguments:
    - data: numpy array of shape (n, 2)
    - labels: numpy array of shape (n,)
    - num_samples: number of samples to extract

    Returns:
    - data_samples: numpy array of shape (num_samples, 2)
    - label_samples: numpy array of shape (num_samples,)
    """
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    data_samples = data[indices]
    label_samples = labels[indices]
    return data_samples, label_samples


# ----------------------------------------------------------------------

def confusion_matrix(true_labels, predicted_labels):
    
    classes = np.unique(np.concatenate((true_labels, predicted_labels)))
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_index = {cls: idx for idx, cls in enumerate(classes)}
    for true, pred in zip(true_labels, predicted_labels):
        conf_matrix[class_index[true]][class_index[pred]] += 1
    return conf_matrix

def em_algorithm(data: NDArray[np.floating], max_iter: int = 100) -> Tuple[Union[NDArray[np.floating], None], 
                                                                         Union[NDArray[np.floating], None], 
                                                                         Union[NDArray[np.floating], None], 
                                                                         Union[NDArray[np.floating], None], 
                                                                         Union[NDArray[np.floating], None]]:
    """
    Arguments:
    - data: numpy array of shape 50,000 x 2
    - max_iter: maximum number of iterations for the algorithm

    Return:
    - weights: the two coefficients \rho_1 and \rho_2 of the mixture model
    - means: the means of the two Gaussians (two scalars) as a list
    - covariances: the covariance matrices of the two Gaussians
      (each is a 2x2 symmetric matrix) return the full matrix
    - log_likelihoods: `max_iter` values of the log_likelihood, including the initial value
    - predicted_labels

    Notes:
    - order the distribution parameters such that the x-component of
          the means are ordered from largest to smallest.
    - hint: the log-likelihood is monotonically increasing (or constant)
          if the algorithm is implemented correctly.
    - If this code is copied from some source, make sure to reference the
        source in this doc-string.
    """
    # CODE FILLED BY STUDENT

    n, d = data.shape

    # Initialize parameters randomly
    weights = np.random.rand(2)
    weights /= np.sum(weights)
    means = np.random.rand(2, d)
    covariances = np.array([np.eye(d)] * 2)

    # log_likelihood
    log_likelihoods = []

    # Probability storage
    probabilities = np.zeros((n, 2))

    for _ in range(max_iter):
        # E-step: update probabilities
        probabilities = np.array(
            [
                multivariate_pdf(data, mean, cov)
                # multivariate_normal.pdf(data, mean, cov)
                for mean, cov in zip(means, covariances)
            ]
        ).T

        # Normalize the probabilities
        responsibilities = probabilities * weights
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # Compute log-likelihood
        weighted_sum = np.sum(probabilities * weights, axis=1)
        log_likelihood = np.sum(np.log(weighted_sum))
        log_likelihoods.append(log_likelihood)

        # M-step: update parameters
        for i in range(2):
            weight = responsibilities[:, i].sum()
            weights[i] = weight / n
            means[i] = (
                np.sum(data * responsibilities[:, i][:, np.newaxis], axis=0) / weight
            )
            cov_diff = data - means[i]
            covariances[i] = (
                np.dot(cov_diff.T, cov_diff * responsibilities[:, i][:, np.newaxis])
                / weight
            )

    # Order the distributions such that the x-component of the means are ordered from largest to smallest
    # print(f"{responsibilities.shape=}")
    if means[0][0] < means[1][0]:
        weights = np.flip(weights)
        means = np.flip(means, axis=0)
        covariances = np.flip(covariances, axis=0)
        responsibilities = np.flip(responsibilities, axis=1)

    # Compute the predicted labels for the probabilities in the correct order.
    # probability 0 has the largeset x component of the mean
    predicted_labels = np.argmax(responsibilities, axis=1)

    return (
        weights,
        means,
        covariances,
        np.array(log_likelihoods),
        predicted_labels,
    )
    # added predicted_labels. Fix type hints in function signature


# ----------------------------------------------------------------------
def gaussian_mixture():
    """
    Calculate the parameters of a Gaussian mixture model using the EM algorithm.
    Specialized to two distributions.
    """
    answers = {}
    # Read data from file and store in a numpy array
    # data file name: "question2_cluster_data.npy"
    # label file name: "question2_cluster_labels.npy"
    data = np.load("question2_cluster_data.npy")
    labels = np.load("question2_cluster_labels.npy")

    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")

    def sample_estimate(data, labels, nb_samples=10000, max_iter=100):
        # ADD STUDENT CODE
        data_samples, label_samples = extract_samples(data, labels, nb_samples)
        weights, means, covariances, log_likelihoods, predicted_labels = em_algorithm(
            data_samples, max_iter
        )
        print("==> means: ", means)
        # Compute the confusion matrix
        # The predicted labels are in the correct order: 0 and 1
        # nb_pred_labels_0 = np.sum(predicted_labels == 0)
        # nb_pred_labels = [nb_pred_labels_0, data_samples.shape[0] - nb_pred_labels_0]
        # nb_data_labels_0 = np.sum(label_samples == 0)
        # nb_data_labels = [nb_data_labels_0, data_samples.shape[0] - nb_data_labels_0]
        """
        if nb_pred_labels_0 > nb_samples // 2 and nb_data_labels_0 < nb_samples // 2:
            # predicted_labels = 1 - predicted_labels
            label_samples = 1 - label_samples
            nb_data_labels_0 = np.sum(label_samples == 0)
        """
        # print("nb_data_labels_0= ", nb_data_labels_0)

        # print("label_samples (1)= ", sum(label_samples == 0))
        # print("predicted_labels= ", sum(predicted_labels == 0))
        # print("means: ", means)
        # print("weights: ", weights)
        # match the labels
        confusion_mat = confusion_matrix(label_samples, predicted_labels)
        # answers["confusion_matrix"] = confusion_mat

        ARI = adjusted_rand_index(label_samples, predicted_labels)
        # ARI_confusion = compute_ARI(confusion_mat)
        # print(f"{ARI=}, {ARI_confusion=}")  # Wrong value (not even close)

        SSE = compute_SSE(data_samples, predicted_labels)

        return weights, means, covariances, log_likelihoods, confusion_mat, ARI, SSE

    # Call the function
    data_samples, label_samples = extract_samples(data, labels, 10000)

    print(f"Data shape: {data_samples.shape}")
    print(f"Labels shape: {label_samples.shape}")

    # ADD STUDENT CODE
    plot = plt.scatter(data_samples[:, 0], data_samples[:, 1], c=label_samples, s=0.1)
    answers["plot_original_cluster"] = plot

    # --------------------------------------------------------------
    # Return the `em_algorithm` funtion
    answers["em_algorithm_function"] = em_algorithm

    # --------------------------------------------------------------
    max_iter = 100
    weights, means, covariances, log_likelihoods, predicted_labels = em_algorithm(
        data, max_iter
    )

    # 1D numpy array of floats
    answers["log_likelihood"] = log_likelihoods

    # --------------------------------------------------------------

    # a line plot using matplotlib.pyplot.plot
    # Make sure to include title, axis labels, and a grid.
    # Save the plot to file "plot_log_likelihood.pdf", and add to your report.
    assert max_iter == log_likelihoods.shape[0]
    plot_likelihood = plt.plot(list(range(max_iter)), log_likelihoods)
    plt.title("Log Likelihood vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.grid(True)
    answers["plot_log_likelihood"] = plot_likelihood
    plt.savefig("plot_log_likelihood.pdf")

    # --------------------------------------------------------------
    # 10 trials of 10,000 points each
    nb_trials = 10
    nb_samples = 10000
    max_iter = max_iter

    means_lst = []
    weights_lst = []
    covariances_lst = []
    confusion_lst = []
    ARI_lst = []
    SSE_lst = []
    for _ in range(nb_trials):
        weights, means, covariances, log_likelihoods, confusion_mat, ARI, SSE = (
            sample_estimate(data, labels, nb_samples=nb_samples, max_iter=max_iter)
        )
        weights_lst.append(weights)
        means_lst.append(means)
        covariances_lst.append(covariances)
        # print("==> confusion_mat: ", confusion_mat)
        confusion_lst.append(np.array(confusion_mat))
        ARI_lst.append(ARI)
        SSE_lst.append(SSE)

    # print("===> list of SSE= ", SSE)
    # Average the results
    weights = np.array(weights_lst)
    means = np.array(means_lst)
    covariances = np.array(covariances_lst)
    avg_weights = np.mean(weights, axis=0)
    avg_means = np.mean(means, axis=0)
    avg_covariances = np.mean(covariances, axis=0)
    std_weights = np.std(weights, axis=0)
    std_means = np.std(means, axis=0)
    std_covariances = np.std(covariances, axis=0)
    avg_ARI = np.mean(ARI_lst)
    std_ARI = np.std(ARI_lst)
    avg_SSE = np.mean(SSE_lst)
    std_SSE = np.std(SSE_lst)

    # list with the mean and standard deviation (over 10 trials) of the mean vector
    # of the first distribution
    answers["probability_1_mean"] = [avg_means[0].tolist(), std_means[0].tolist()]
    # print("answers probability_1_mean", answers["probability_1_mean"])
    answers["probability_2_mean"] = [avg_means[0].tolist(), std_means[1].tolist()]

    # list with the mean and standard deviation (over 10 trials) of the covariance matrix
    # of the first distribution. The covariance matrix should be in the standard order.
    # (https://www.cuemath.com/algebra/covariance-matrix/)
    #
    answers["probability_1_covariance"] = [
        avg_covariances[0],
        std_covariances[0],
    ]
    answers["probability_2_covariance"] = [
        avg_covariances[1],
        std_covariances[1],
    ]

    # print(f"{answers['probability_1_covariance']=}")
    # print(f"{answers['probability_2_covariance']=}")

    # list with the mean and standard deviation (over 10 trials) of the amplitude \rho_1
    # of the first distribution.
    answers["probability_1_amplitude"] = [
        avg_weights[0].tolist(),
        std_weights[0].tolist(),
    ]
    answers["probability_2_amplitude"] = [
        avg_weights[1].tolist(),
        std_weights[1].tolist(),
    ]

    # Return a 2x2 numpy Array of floats. Average of the confusion matrices of the 10 trials
    answers["average_confusion_matrix"] = np.mean(confusion_lst, axis=0)

    # Return a 2x2 numpy Array of floats. Standard deviation of the confusion matrices of the 10 trials
    # This means to take the standard deviation of each element of the confusion matrix.
    # So there are 10 (1,1) elements, and you are to average these and take the standard deviation.
    answers["std_confusion_matrix"] = np.std(confusion_lst, axis=0)

    # Return a list of 10 ARIs (float), one for each sample of 5,000 points
    answers["ARI"] = ARI_lst

    # Return a list of 10 SSEs (float), one for each sample of 5,000 points
    answers["SSE"] = SSE_lst
    # print(f"{SSE_lst=}")

    # Return the mean and standard deviation of the 10 trials (of 5000 points each)
    answers["avg_std_ARI"] = [avg_ARI, std_ARI]
    # print(f"{avg_ARI=}, {std_ARI=} ")

    # Return the mean and standard deviation of the 10 trials (of 5000 points each)
    answers["avg_std_SSE"] = [avg_SSE, std_SSE]
    # print(f"{avg_SSE=}, {std_SSE=}")

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = gaussian_mixture()
    # del all_answers["em_algorithm_function"]  # = None

    with open("gaussian_mixture.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
