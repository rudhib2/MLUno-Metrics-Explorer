import matplotlib.pyplot as plt
from mluno.data import make_line_data
from mluno.plot import plot_predictions
from sklearn.linear_model import LinearRegression


def test_plot_predictions():
    X, y = make_line_data()
    regressor = LinearRegression()
    regressor.fit(X, y)

    fig, ax = plot_predictions(X, y, regressor, title="Test Plot")

    # Check that the function returns a Figure and Axes object
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Check that the title of the plot is correct
    assert ax.get_title() == "Test Plot"