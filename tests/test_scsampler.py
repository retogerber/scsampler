import unittest
from unittest.mock import patch

import numpy as np
from anndata import AnnData

from scsampler import scsampler


class ScSamplerTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [3.0, 3.0],
                [3.0, 4.0],
            ]
        )

    def test_numpy_copy_returns_matrix_and_indices(self):
        subset, indices = scsampler(self.matrix, n_obs=3, copy=True, random_state=0, selection_method="exact")
        self.assertEqual(subset.shape, (3, 2))
        self.assertEqual(indices.shape, (3,))
        self.assertEqual(len(np.unique(indices)), 3)

    def test_neighbor_sampling_returns_unique_indices(self):
        indices = scsampler(self.matrix, n_obs=4, random_state=1, selection_method="neighbors", n_neighbors=2)
        self.assertEqual(indices.shape, (4,))
        self.assertEqual(len(np.unique(indices)), 4)

    def test_anndata_round_trip(self):
        adata = AnnData(X=self.matrix.copy())
        adata.obsm["X_pca"] = self.matrix.copy()
        subset = scsampler(adata, fraction=0.5, copy=True, random_state=2, selection_method="neighbors", n_neighbors=2)
        self.assertEqual(subset.n_obs, 3)

    def test_invalid_fraction_raises(self):
        with self.assertRaises(ValueError):
            scsampler(self.matrix, fraction=1.5)

    def test_gpu_backend_requires_optional_dependency(self):
        with patch("scsampler.backends.cp", None):
            with self.assertRaises(ImportError):
                scsampler(self.matrix, n_obs=2, backend="gpu", selection_method="exact")


if __name__ == "__main__":
    unittest.main()
