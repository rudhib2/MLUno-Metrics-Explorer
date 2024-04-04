import numpy as np

class KNNRegressor:
    """

    A class used to represent a K-Nearest Neighbors Regressor.

    Parameters
    ----------
    k : int
        The number of nearest neighbors to consider for regression.

    """

    def __init__(self, k=5):
        """
        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors to consider for regression. Default is 5.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using X as input data and y as target values.

        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape (n_samples, 1) where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape (n_samples, ).
        """

        self.X = X
        self.y = y

    def __repr__(self) -> str:
        """
        KNNRegressor.__repr__

        Returns a string representation of the KNNRegressor object.

        Returns
        -------
        str
            A string containing information about the KNNRegressor object.
        """

        return f"KNN Regression model with k = {self.k}."

    def predict(self, X_new):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X_new : ndarray
            Input data, a 2D array of shape (n_samples, 1), with which to make predictions.

        Returns
        -------
        ndarray
            The target values, which is a 1D array of shape (n_samples, ).
        """

        predicted_labels = [self._predict(x) for x in X_new]
        return np.array(predicted_labels)

    def _predict(self, x_new):
        """
        KNNRegressor._predict

        Predict the target for a single data point.

        Parameters
        ----------
        x_new : ndarray
            Input data, a 1D array representing a single data point.

        Returns
        -------
        float
            The predicted target value for the given data point.
        """

        # compute distances between new x and all samples in the X data
        distances = [np.linalg.norm(x_new - x) for x in self.X]
        # sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[: self.k]
        # extract the labels of the k nearest neighbor training samples
        k_nearest_y = self.y[k_indices]
        # return the mean of the k nearest neighbors
        return np.mean(k_nearest_y)


class LinearRegressor:
    """
    A class used to represent a Simple Linear Regressor.

    Attributes
    ----------
    weights : ndarray
        The weights of the linear regression model. Here, the weights are represented by the β
        vector which for univariate regression is a 1D vector of length two, β = [β0, β1], where β0 is the slope and β1 is the intercept.

    """

    def __init__(self):
        """
        Initializes the LinearRegressor.
        """
        self.weights = None

    def fit(self, X, y):
        """
        Trains the linear regression model using the given training data.

        In other words, the fit method learns the weights, represented by the β
        vector. To learn the β vector, use:

        `B̂ = np.linalg.inv(X.T @ X) @ X.T @ y`

        Here, 
        is the so-called design matrix, which, to include a term for the intercept, has a column of ones appended to the input X matrix.

        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape (n_samples, 1) where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape (n_samples, ).

        """
        ones_column = np.ones((X.shape[0], 1))
        design_matrix = np.hstack((ones_column, X))
        self.weights = np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y

    def predict(self, X):
        """
        Makes predictions for input data.

        Parameters
        ----------
        X : ndarray
            Input data, a 2D array of shape (n_samples, 1), with which to make predictions.

        Returns
        -------
        ndarray
            The predicted target values as a 1D array with the same length as X.
        """
        ones_column = np.ones((X.shape[0], 1))
        design_matrix = np.hstack((ones_column, X))
        return design_matrix @ self.weights