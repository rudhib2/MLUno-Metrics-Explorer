import numpy as np

def make_line_data(n_samples=100, beta_0=0, beta_1=1, sd=1, X_low=-10, X_high=10, random_seed=None):
    """
    Generate data for linear regression.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate. 
    beta_0 : float
        The true intercept of the linear model. 
    beta_1 : float
        The true slope of the linear model. 
    sd : float
        Standard deviation of the normally distributed errors.
    X_low : float
        Lower bound for the uniform distribution of X.
    X_high : float
        Upper bound for the uniform distribution of X.
    random_seed : int
        Seed to control randomness.
    
    Returns
    -------
    tuple
        A tuple containing the X and y arrays. X is a 2D array with shape (n_samples, 1) and y is a 1D array with shape (n_samples,). X contains the simulated X values and y contains the corresponding true mean of the linear model with added normally distributed errors.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    X = np.random.uniform(low=X_low, high=X_high, size=(n_samples, 1))
    y = beta_0 + beta_1 * X.ravel() + np.random.normal(scale=sd, size=n_samples)
    return X, y


def make_sine_data(n_samples=100, sd=1, X_low=-6, X_high=6, random_seed=None):
    """
    Generate data for nonlinear regression.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    sd : float
        Standard deviation of the normally distributed errors.
    X_low : float
        Lower bound for simulated X values.
    X_high : float
        Upper bound for simulated X values.
    random_seed : int
        Seed to control randomness.
    
    Returns
    -------
    tuple
        A tuple containing the X and y arrays. X is a 2D array with shape (n_samples, 1) and y is a 1D array with shape (n_samples,). 
        X contains the simulated X values and y contains the corresponding sine values with added normally distributed errors.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    X = np.random.uniform(low=X_low, high=X_high, size=(n_samples, 1))
    y = np.sin(X).ravel() + np.random.normal(scale=sd, size=n_samples)
    return X, y

def split_data(X, y, holdout_size=0.2, random_seed=None):
    """
    Split the data into train and test sets.

    Parameters
    ----------
    X : ndarray
        The feature data to be split. A 2D array with shape (n_samples, 1).
    y : ndarray
        The target data to be split. A 1D array with shape (n_samples,).
    holdout_size : float
        The proportion of the data to be used as the test set.
    random_seed : int
        Seed to control randomness.
    
    Returns
    -------
    tuple
        The split train and test data: (X_train, X_test, y_train, y_test).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # Shuffle indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    # Calculate split index
    split_index = int(X.shape[0] * (1 - holdout_size))
    # Split data
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
    return X_train, X_test, y_train, y_test
