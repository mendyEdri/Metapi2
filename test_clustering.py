import numpy as np
from matplotlib.figure import Figure

from clustering import visualize_clusters


def test_visualize_clusters_adjusts_perplexity():
    """visualize_clusters should adapt t-SNE perplexity for tiny datasets."""
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = [0, 1]
    fig = visualize_clusters(embeddings, labels)
    assert isinstance(fig, Figure)
