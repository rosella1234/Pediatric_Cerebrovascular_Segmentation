import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import csv
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd

from tof_mra.common import read_cases, load_config, write_cases, write_json
from tof_mra.comparison import compare_reports, merge_reports
from tof_mra.conversion import convert_series
from tof_mra.dataset import prepare_study, validate_splits
from tof_mra.evaluation import evaluate
from tof_mra.imaging import image, prepare_label, require_geometry, binary_data
from tof_mra.nnunet import predict_command, train_command


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, name, array=None, affine=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8) if array is None else array,
                              np.eye(4) if affine is None else affine), path)
        return path

    def config(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "study.example.json"
        cfg = json.loads(path.read_text())
        cfg["manifest"] = "cases.csv"
        cfg["work_dir"] = "work"
        for key in ("raw", "preprocessed", "results"):
            cfg["nnunet"][key] = "work/" + key
        cfg["baseline"]["results"] = "baseline_models"
        cfg["dataset"]["expected_development"] = 1
        cfg["dataset"]["expected_test"] = 1
        cfg["dataset"]["expected_validation"] = None
        cfg["evaluation"]["expected_pairs"] = 1
        write_json(self.root / "study.json", cfg)
        return load_config(self.root / "study.json")

    def cases(self):
        rows = []
        for case, split in (("dev001", "development"), ("test001", "test")):
            rows.append({"case_id": case, "subject_id": case, "split": split,
                         "tof": str(self.save(case + "_image.nii.gz")), "label": str(self.save(case + "_label.nii.gz")),
                         "brain_mask": str(self.save(case + "_brain.nii.gz")), "t1": ""})
        write_cases(self.root / "cases.csv", rows)
        return rows

    def reports(self, *, identical=False):
        rows = []
        for i in range(6):
            rows.append({"case_id": f"case{i}", "Dice": .7 + .01*i, "Recall": .6 + .01*i,
                         "Precision": .8 + .01*i, "IoU": .6 + .01*i, "HD95_mm": 2 + i/10,
                         "AHD_mm": .3 + i/100, "Balanced_AHD_reference_mm": .4 + i/100,
                         "Balanced_AHD_prediction_mm": .5 + i/100})
        base = pd.DataFrame(rows)
        trained = base.copy()
        if not identical:
            for key in ("Dice", "Recall", "Precision", "IoU"):
                trained[key] += np.arange(1, 7)/100
            for key in ("HD95_mm", "AHD_mm", "Balanced_AHD_reference_mm", "Balanced_AHD_prediction_mm"):
                trained[key] *= np.linspace(.5, .7, 6)
        base.to_csv(self.root / "baseline.csv", index=False)
        trained.iloc[::-1].to_csv(self.root / "trained.csv", index=False)
        return self.root / "baseline.csv", self.root / "trained.csv"

    def test_case_id_pairing_is_independent_of_row_order(self):
        b, t = self.reports()
        result = compare_reports(b, t, self.root / "comparison", expected_pairs=6, make_plots=False)
        self.assertTrue((result.n_pairs == 6).all())
        self.assertAlmostEqual(result.iloc[0].p_value, .03125)
        self.assertAlmostEqual(result.iloc[0].median_paired_change_trained_minus_baseline, .035)

    def test_all_zero_paired_differences(self):
        b, t = self.reports(identical=True)
        result = compare_reports(b, t, self.root / "comparison", make_plots=False)
        self.assertTrue((result.p_value == 1).all())

    def test_missing_pair_is_rejected(self):
        b, t = self.reports()
        pd.read_csv(t).iloc[:-1].to_csv(t, index=False)
        with self.assertRaisesRegex(ValueError, "identical case sets"):
            compare_reports(b, t, self.root / "comparison", make_plots=False)

    def test_retained_subset_does_not_reuse_full_cohort_statistics(self):
        b, t = self.reports()
        with self.assertRaisesRegex(ValueError, "Expected 19"):
            compare_reports(b, t, self.root / "comparison", expected_pairs=19, make_plots=False)

    def test_duplicate_report_id_is_rejected(self):
        b, t = self.reports()
        frame = pd.read_csv(t)
        frame.loc[1, "case_id"] = frame.loc[0, "case_id"]
        frame.to_csv(t, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_reports(b, t, self.root / "comparison", make_plots=False)

    def test_nonfinite_values_require_explicit_exclusion(self):
        b, t = self.reports()
        frame = pd.read_csv(t)
        frame.loc[0, "AHD_mm"] = np.inf
        frame.to_csv(t, index=False)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compare_reports(b, t, self.root / "comparison", make_plots=False)
        result = compare_reports(b, t, self.root / "comparison", make_plots=False, nonfinite_policy="drop")
        self.assertEqual(result.set_index("metric").loc["AHD_mm", "n_pairs"], 5)
        audit = json.loads((self.root / "comparison" / "pairing_audit.json").read_text())
        self.assertEqual(len(audit["excluded_nonfinite_by_metric"]["AHD_mm"]), 1)

    def test_historical_bahd_is_never_relabelled_as_reference_normalized(self):
        b, t = self.reports()
        for path in (b, t):
            frame = pd.read_csv(path).drop(columns="Balanced_AHD_reference_mm").rename(columns={"Balanced_AHD_prediction_mm": "Balanced_AHD_mm"})
            frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Historical"):
            compare_reports(b, t, self.root / "comparison", make_plots=False)
        result = compare_reports(b, t, self.root / "comparison", make_plots=False, bahd_normalization="prediction")
        self.assertIn("Balanced_AHD_prediction_mm", result.metric.tolist())

    def test_different_brain_mask_policies_are_rejected(self):
        b, t = self.reports()
        for path, policy in ((b, "required"), (t, "none")):
            frame = pd.read_csv(path)
            frame["brain_mask_policy"] = policy
            frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "different brain-mask policies"):
            compare_reports(b, t, self.root / "comparison", make_plots=False)

    def test_geometry_checks_include_affine(self):
        a = self.save("a.nii.gz")
        affine = np.eye(4)
        affine[0, 3] = 4
        b = self.save("b.nii.gz", affine=affine)
        with self.assertRaisesRegex(ValueError, "Geometry mismatch"):
            require_geometry(image(a), image(b), context="same shape, different origin")

    def test_nearest_neighbor_label_alignment(self):
        reference = self.save("reference.nii.gz")
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1, 2, 2] = 1
        affine = np.eye(4)
        affine[0, 3] = 1
        label = self.save("source_label.nii.gz", mask, affine)
        target = self.root / "aligned.nii.gz"
        record = prepare_label(label, reference, target, resample=True)
        self.assertTrue(record["resampled"])
        result = np.asanyarray(image(target).dataobj)
        self.assertEqual(result[2, 2, 2], 1)
        self.assertEqual(int(result.sum()), 1)

    def test_probability_maps_are_rejected_as_binary_labels(self):
        path = self.save("probability.nii.gz", np.full((5, 5, 5), .25, dtype=np.float32))
        with self.assertRaises(ValueError):
            binary_data(image(path))

    def test_preparation_keeps_test_labels_out_of_raw_dataset(self):
        self.cases()
        cfg = self.config()
        path = prepare_study(cfg)
        self.assertEqual([p.name for p in (path / "labelsTr").glob("*.nii.gz")], ["dev001.nii.gz"])
        self.assertTrue((self.root / "work" / "test_inputs" / "test001_0000.nii.gz").is_file())
        self.assertFalse((path / "labelsTs").exists())
        self.assertEqual(json.loads((path / "dataset.json").read_text())["channel_names"], {"0": "CT"})

    def test_subject_reuse_and_saved_fold_leakage_are_rejected(self):
        rows = self.cases()
        rows[1]["subject_id"] = rows[0]["subject_id"]
        write_cases(self.root / "cases.csv", rows)
        with self.assertRaisesRegex(ValueError, "Repeated subject_id"):
            read_cases(self.root / "cases.csv")
        with self.assertRaisesRegex(ValueError, "partition"):
            validate_splits([{"train": ["dev001"], "val": ["test001"]}], {"dev001", "dev002"}, self.config())

    def test_brain_mask_is_applied_to_both_masks(self):
        rows = self.cases()
        cfg = self.config()
        ref = np.zeros((5, 5, 5), dtype=np.uint8)
        ref[2, 2, 2] = 1
        pred = ref.copy()
        pred[0, 0, 0] = 1
        brain = np.ones_like(ref)
        brain[0, 0, 0] = 0
        self.save("test001_label.nii.gz", ref)
        self.save("test001_brain.nii.gz", brain)
        pred_dir = self.root / "predictions"
        self.save("predictions/test001.nii.gz", pred)
        csv_path = evaluate(cfg, "baseline", pred_dir)
        self.assertEqual(pd.read_csv(csv_path).iloc[0].Dice, 1)

    def test_commands_preserve_fold_checkpoint_and_no_weight_initialization(self):
        cfg = self.config()
        baseline = predict_command(cfg, "baseline")
        pediatric = predict_command(cfg, "pediatric")
        self.assertEqual(baseline[baseline.index("-c")+1], "2d")
        self.assertEqual(pediatric[pediatric.index("-c")+1], "3d_fullres")
        self.assertEqual(baseline[baseline.index("-f")+1], "0")
        self.assertIn("checkpoint_final.pth", pediatric)
        self.assertNotIn("--disable_tta", baseline)
        self.assertNotIn("-pretrained_weights", train_command(cfg))

    def test_dicom_decompression_fallback(self):
        series = self.root / "dicom"
        series.mkdir()
        (series / "slice.dcm").write_bytes(b"synthetic placeholder")
        manifest = self.root / "series.csv"
        manifest.write_text("case_id,modality,series_dir\ncase001,tof,dicom\n")
        attempts = []

        def fake_run(argv, **kwargs):
            attempts.append(argv)
            if argv[0] == "decoder":
                Path(argv[-1]).write_bytes(b"decompressed placeholder")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if len(attempts) == 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="JPEG decoder unavailable")
            out = Path(argv[argv.index("-o")+1]) / "series_1.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4)), out)
            return SimpleNamespace(returncode=0, stdout="converted", stderr="")

        with patch("tof_mra.conversion.subprocess.run", side_effect=fake_run):
            result = convert_series(manifest, self.root / "converted", decompressor=["decoder"])
        self.assertTrue(result[0]["decompression_used"])
        self.assertTrue((self.root / "converted" / "tof" / "case001.nii.gz").is_file())
        self.assertTrue((series / "slice.dcm").is_file())


if __name__ == "__main__":
    unittest.main()

