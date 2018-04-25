"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal
# from scipy.cluster.vq import kmeans2


class GaussianMixtureModel(object):
    """Gaussian Mixture Model."""

    def __init__(self, n_dims, n_components=1,
                 max_iter=25,
                 reg_covar=1e-6):
        """
        Gaussian Mixture Model init funtion.

        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        # np.array of size (n_components, n_dims)
        self._mu = np.random.rand(self._n_components, self._n_dims)

        # Initialized with uniform distribution.
        # np.array of size (n_components, 1)
        tmp = np.random.dirichlet(np.ones(self._n_components), size=1)
        self._pi = tmp.reshape(tmp.shape[1], 1)

        # Initialized with identity.
        # np.array of size (n_components, n_dims, n_dims)
        i = np.eye(self._n_dims) * 100
        self._sigma = np.repeat(i[np.newaxis, :, :],
                                self._n_components, axis=0)

    def fit(self, x):
        """Run EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        # self._mu, _ = kmeans2(x, self._n_components)
        self._mu = x[np.random.choice(
            x.shape[0], size=self._n_components, replace=False), :]

        for iters in range(self._max_iter):
            z_ik = self._e_step(x)
            # print("e_step: ", iters)
            self._m_step(x, z_ik)
            # print("m_step: ", iters)

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        # Update for pi (n_components, 1)

        # avg = np.mean(z_ik, axis=0).tolist()
        # norm = [i / sum(avg) for i in avg]
        # self._pi = np.array(norm).reshape(-1, 1)

        sum_ = np.sum(z_ik, axis=0)
        self._pi = sum_ / x.shape[0]

        # Update for mu (n_components, ndims)
        # new_mu = np.zeros_like(self._mu)
        # mu_down = np.sum(z_ik, axis=0)
        # for k in range(self._n_components):
        #     mu_up = np.zeros((1, self._n_dims))
        #     for i in range(x.shape[0]):
        #         mu_up += z_ik[i, k] * x[i, :]
        #     new_mu[k, :] = mu_up / mu_down[k]

        mu_up = z_ik.T.dot(x)
        mu_down = np.sum(z_ik, axis=0).reshape(-1, 1)
        self._mu = mu_up / mu_down

        # Update for sigma (n_components, n_dims, n_dims)
        new_sigma = np.zeros_like(self._sigma)
        sigma_down = np.sum(z_ik, axis=0)
        reg = np.zeros((self._n_dims, self._n_dims))
        np.fill_diagonal(reg, self._reg_covar)

        for k in range(self._n_components):
            # mu_k = self._mu[k, :]
            # sigma_k = np.zeros((self._n_dims, self._n_dims))
            # for i in range(x.shape[0]):
            #     sigma_k += z_ik[i, k] * np.diag(self._reg_covar +
            #                                     np.diag(np.outer(x[i, :]
            #                                    - mu_k, x[i, :] - mu_k)))
            # new_sigma[k] = sigma_k / sigma_down[k]
            x_demean = x - self._mu[k, :]
            sigma_up = z_ik[:, k][:, np.newaxis] * x_demean
            new_sigma[k, :, :] = x_demean.T.dot(sigma_up) / sigma_down[k] + reg
        self._sigma = new_sigma

    def get_conditional(self, x):
        """Compute the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            response(numpy.ndarray): The conditional probability for each example,
                dimension (N, n_components).
        """
        # response = np.zeros((x.shape[0], self._n_components))
        response = []

        # compute conditional probability for each data example
        for k in range(self._n_components):
            # for i in range(x.shape[0]):
                # response[i, k] = self._multivariate_gaussian(
                #     x[i, :], self._mu[k, :], self._sigma[k])
            response.append(self._multivariate_gaussian(
                x, self._mu[k], self._sigma[k]))

        response = np.transpose(np.array(response))
        return response

    def get_marginals(self, x):
        """Compute the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            The marginal probability for each example, dimension (N,).
        """
        # get conditional probability
        conditions = self.get_conditional(x)

        # multiply conditional probability with pi_{k}
        culmulate = conditions.dot(self._pi)
        return culmulate.flatten()

    def get_posterior(self, x):
        """Compute the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        # get conditional probability
        conditions = self.get_conditional(x)

        # get marginal probability
        marginals = self.get_marginals(x)

        # for i in range(conditions.shape[0]):
        #     down = marginals[i] + np.finfo(float).eps
        #     for k in range(self._n_components):
        #         up = conditions[i, k] * self._pi[k]
        #         z_ik[i, k] = up / down

        weighted_conditions = np.multiply(conditions, np.transpose(self._pi))
        z_ik = np.transpose(np.transpose(
            weighted_conditions) / (marginals + np.finfo(float).eps))
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.

        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,)
            sigma_k(numpy.ndarray): Array containing one signle covariance
                matrix (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.

        For each cluster, find the most common digit using the provided (x, y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.cluster_label_map = np.random.rand(self._n_components).tolist()

        # Perform EM in dataset
        self.fit(x)

        # Get z_{ik} after performing EM algorithm
        z_ik = self.get_posterior(x)

        # Get the highest probability of k^th GMM for each data points
        # Assign labels for data points
        em_label = np.argmax(z_ik, axis=1)

        # Check with grountruth labels
        for k in range(self._n_components):
            data_idx = np.where(em_label == k)
            if data_idx[0].size:
                coor_data = y[data_idx]
                vote_label = scipy.stats.mode(coor_data)[0][0]
                self.cluster_label_map[k] = vote_label

    def supervised_predict(self, x):
        """Predict a label for each example in x.

        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """
        z_ik = self.get_posterior(x)
        em_label = np.argmax(z_ik, axis=1)
        y_hat = [self.cluster_label_map[idx] for idx in em_label]
        return np.array(y_hat)
