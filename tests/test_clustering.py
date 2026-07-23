import unittest

import numpy as np

import clustering


class ClusteringTests(unittest.TestCase):
    def test_squared_euclidean_distance(self):
        result = clustering.squared_euclidean_distance([0, 0], [3, 4])
        self.assertEqual(result, 25.0)

    def test_distance_matrix_is_symmetric(self):
        data = np.array([[0, 0], [3, 4], [0, 4]], dtype=float)
        matrix = clustering.calculate_distance_matrix(data, "euclidean")
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), np.zeros(3))

    def test_complete_linkage_separates_two_obvious_groups(self):
        data = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
        labels = clustering.complete_linkage_clustering(data, n_clusters=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_invalid_cluster_count_is_rejected(self):
        with self.assertRaises(ValueError):
            clustering.complete_linkage_clustering([[0, 0]], n_clusters=2)

    def test_invalid_distance_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            clustering.calculate_distance_matrix([[0, 0]], "manhattan")


if __name__ == "__main__":
    unittest.main()
