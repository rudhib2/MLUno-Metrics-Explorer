import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(X, y, regressor, conformal=False, title=''):
    """
    Plot predictions of a regressor along with the data.

    Parameters
    ----------
    X : ndarray
        The input data to the regressor. A 2D array.
    y : ndarray
        The target values. A 1D array of the same length as X.
    regressor : object
        The regressor object. It should have a predict method that accepts X and returns predictions.
    conformal : bool, optional
        If True, the regressor is assumed to return prediction intervals (lower and upper bounds) along with the predictions.
        The prediction intervals are plotted as a shaded area. Default is False.
    title : str, optional
        The title of the plot. Default is an empty string.

    Returns
    -------
    ``` matplotlib.figure.Figure ```
        The figure object of the plot.
    ``` matplotlib.axes.Axes ```
        The axes object of the plot.

    Notes
    -----
    This function assumes that the predict method of regressor returns a tuple of three elements (predictions, lower bounds, upper bounds) when conformal is True.
    """
    if conformal:
        predictions, lower_bounds, upper_bounds = regressor.predict(X)
    else:
        predictions = regressor.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Data')
    ax.plot(X, predictions, label='Predictions', color='red')

    if conformal:
        ax.fill_between(X.ravel(), lower_bounds, upper_bounds, color='gray', alpha=0.2, label='Prediction Intervals')

    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()

    return fig, ax