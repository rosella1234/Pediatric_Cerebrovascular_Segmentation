import unittest
import numpy as np

from tof_mra.metrics import distance_metrics, overlap_metrics, surface_points


class MetricTests(unittest.TestCase):
    def test_overlap_hand_calculation(self):
        scores = overlap_metrics(np.array([1, 1, 0]), np.array([1, 0, 1]))
        for name in ("Dice", "Recall", "Precision", "F1_Score"):
            self.assertEqual(scores[name], 0.5)
        self.assertAlmostEqual(scores["IoU"], 1/3)

    def test_empty_masks_match_source_convention(self):
        empty = np.zeros((4, 4, 4), dtype=bool)
        self.assertTrue(all(x == 1 for x in overlap_metrics(empty, empty).values()))
        result = distance_metrics(empty, empty, np.eye(4))
        self.assertTrue(np.isinf(result["AHD_mm"]))
        self.assertEqual(result["distance_status"], "both_empty")
        point = empty.copy()
        point[1, 1, 1] = True
        self.assertEqual(overlap_metrics(empty, point)["Dice"], 0)
        self.assertEqual(distance_metrics(empty, point, np.eye(4))["distance_status"], "one_empty")

    def test_balanced_denominators_are_distinct(self):
        pred = np.zeros((5, 5, 5), dtype=bool)
        ref = pred.copy()
        pred[1:3, 1, 1] = True
        ref[1, 1, 1] = True
        result = distance_metrics(pred, ref, np.diag([2., 1., 1., 1.]))
        self.assertAlmostEqual(result["HD95_mm"], 1.9)
        self.assertAlmostEqual(result["AHD_mm"], 0.5)
        self.assertAlmostEqual(result["Balanced_AHD_reference_mm"], 1.0)
        self.assertAlmostEqual(result["Balanced_AHD_prediction_mm"], 0.5)

    def test_anisotropic_physical_distance(self):
        pred = np.zeros((5, 5, 5), dtype=bool)
        ref = pred.copy()
        pred[1, 1, 1], ref[2, 1, 1] = True, True
        result = distance_metrics(pred, ref, np.diag([2.5, 0.5, 0.7, 1]))
        for key in ("HD95_mm", "AHD_mm", "Balanced_AHD_reference_mm", "Balanced_AHD_prediction_mm"):
            self.assertAlmostEqual(result[key], 2.5)

    def test_rigid_rotation_and_translation_preserve_distances(self):
        pred = np.zeros((5, 5, 5), dtype=bool)
        ref = pred.copy()
        pred[1, 1, 1], ref[2, 2, 1] = True, True
        affine = np.array([[0., -0.5, 0, 123], [2.5, 0, 0, -50], [0, 0, 0.7, 21], [0, 0, 0, 1]])
        self.assertAlmostEqual(distance_metrics(pred, ref, affine)["AHD_mm"], np.hypot(2.5, 0.5))

    def test_surface_is_not_the_filled_volume(self):
        cube = np.ones((3, 3, 3), dtype=bool)
        self.assertEqual(len(surface_points(cube, np.eye(4))), 26)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            overlap_metrics(np.zeros((2, 3)), np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()

