import numpy as np
from matplotlib.figure import Figure

from clustering import visualize_clusters


def test_visualize_clusters_uses_pca_for_small_datasets():
    """visualize_clusters should fall back to PCA when samples are ≤3."""
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = [0, 1]
    fig = visualize_clusters(embeddings, labels)
    assert isinstance(fig, Figure)
    assert "PCA" in fig.axes[0].get_title()


def test_visualize_clusters_clamps_perplexity():
    """t-SNE perplexity is clamped to ``n_samples - 1`` when too large."""
    embeddings = np.array([[float(i), float(i)] for i in range(5)])
    labels = [0, 1, 0, 1, 0]
    fig = visualize_clusters(embeddings, labels)
    assert "perp=4.0" in fig.axes[0].get_title()
