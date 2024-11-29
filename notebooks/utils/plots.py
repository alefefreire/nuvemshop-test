import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from scipy import stats


def plot_comparative_histograms(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    n_rows: int,
    n_cols: int,
    color1: str = "blue",
    color2: str = "red",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    figsize: t.Tuple[int, int] = (20, 15),
    bins: int = 50,
    alpha: float = 0.5,
):
    """
    Plots side-by-side comparative histograms for features in two DataFrames.

    Args:
        dataset1: First DataFrame
        dataset2: Second DataFrame
        n_rows: Number of rows in subplot grid
        n_cols: Number of columns in subplot grid
        color1: Color for first dataset
        color2: Color for second dataset
        label1: Label for first dataset
        label2: Label for second dataset
        figsize: Figure size
        bins: Number of histogram bins
        alpha: Transparency of histograms
    """
    # Get common features between datasets
    common_features = list(set(dataset1.columns) & set(dataset2.columns))

    # Adjust figure size to accommodate side-by-side plots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols * 2, figsize=figsize)

    for idx, feature in enumerate(common_features):
        row, col = divmod(idx, n_cols)
        base_col = col * 2

        # Plot first dataset
        axes[row, base_col].hist(
            dataset1[feature],
            bins=bins,
            color=color1,
            alpha=alpha,
            label=label1,
        )
        axes[row, base_col].set_title(f"{feature}")
        axes[row, base_col].legend()
        axes[row, base_col].grid(axis="y")

        # Plot second dataset
        axes[row, base_col + 1].hist(
            dataset2[feature],
            bins=bins,
            color=color2,
            alpha=alpha,
            label=label2,
        )
        axes[row, base_col + 1].set_title(f"{feature}")
        axes[row, base_col + 1].legend()
        axes[row, base_col + 1].grid(axis="y")

    # Remove empty plots
    for ax in axes.ravel()[len(common_features) * 2 :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_corr_ellipses(data, figsize, **kwargs):
    """
    Plots a correlation matrix using ellipses to represent the correlations.

    Parameters:
    - data (pd.DataFrame): A 2D array or DataFrame containing the correlation matrix.
    - figsize: Tuple specifying the figure size.
    - kwargs: Additional keyword arguments for EllipseCollection.

    Returns:
    - A tuple containing the EllipseCollection object and the Axes object.

    """
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError("Data must be a 2D array.")

    # Mask the upper triangle of the matrix
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    M[mask] = np.nan

    # Initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"aspect": "equal"})
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(-0.5, M.shape[0] - 0.5)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.grid(False)

    # Determine xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # Define ellipse properties
    w = np.ones_like(M).ravel() + 0.01  # Widths of ellipses
    h = 1 - np.abs(M).ravel() - 0.01  # Heights of ellipses
    a = 45 * np.sign(M).ravel()  # Rotation angles

    # Create and add the ellipse collection
    ec = EllipseCollection(
        widths=w,
        heights=h,
        angles=a,
        units="x",
        offsets=xy,
        norm=Normalize(vmin=-1, vmax=1),
        transOffset=ax.transData,
        array=M.ravel(),
        **kwargs,
    )
    ax.add_collection(ec)

    # Add a color bar for correlation values
    cb = fig.colorbar(ec, ax=ax, orientation="horizontal", fraction=0.047, pad=0.00)
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(top=False, labeltop=False)

    # Feature names on the diagonal
    if isinstance(data, pd.DataFrame):
        diagonal_positions = np.arange(M.shape[1])
        for i, label in enumerate(data.columns):
            ax.annotate(
                " -  " + label, (i - 0.4, i - 1), ha="left", va="bottom", rotation=0
            )
        ax.set_yticks(diagonal_positions)
        ax.set_yticklabels(data.index)

    # Hide the plot spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ec, ax


def qq_plot(
    qq_df: pd.DataFrame, n_rows: int, n_cols: int, figsize: t.Tuple[int, int] = (15, 15)
):
    features = qq_df.columns

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    for idx, feature in enumerate(features):
        # Identify row and column in the grid
        row, col = divmod(idx, 3)
        stats.probplot(qq_df[feature], dist="norm", plot=axes[row, col], rvalue=True)
        axes[row, col].set_title(f"QQ plot {feature}")

        axes[row, col].set_xlabel("Theoretical quantiles")
        axes[row, col].set_ylabel("Ordered values")

    # Remove empty plots
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    # Adjust layout
    plt.tight_layout()
    plt.show()
