import numpy as np
from matplotlib.figure import Figure

from clustering import visualize_clusters


def test_visualize_clusters_uses_pca():
    """visualize_clusters should always use PCA for dimensionality reduction."""
    embeddings = np.array([[float(i), float(i)] for i in range(5)])
    labels = [0, 1, 0, 1, 0]
    fig = visualize_clusters(embeddings, labels)
    assert isinstance(fig, Figure)
    assert "PCA" in fig.axes[0].get_title()
