import numpy as np

from clustering import geometric_median, compute_prompt_center


def test_geometric_median_robust_to_outlier():
    # Three identical points and one outlier in the opposite direction.
    points = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    median = geometric_median(points)
    assert np.allclose(median, np.array([1.0, 0.0]), atol=1e-3)


def test_compute_prompt_center_returns_angles():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    center, radial = compute_prompt_center(embeddings)
    # center should be unit norm and close to the majority direction
    assert np.isclose(np.linalg.norm(center), 1.0)
    assert np.allclose(center, np.array([1.0, 0.0]), atol=1e-3)
    # three points aligned with center -> angle 0, outlier -> angle pi
    assert radial.shape == (4,)
    assert np.allclose(radial[:3], 0.0, atol=1e-6)
    assert np.isclose(radial[3], np.pi, atol=1e-6)
