import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from tof_mra.caravel_data import FEATURES, ROOTS, normalize_summary, aggregate_whole_brain, weighted_mean, import_demographics
from tof_mra.caravel_stats import fdr_bh, balance_sexes, sex_comparisons, regional_statistics, age_correlations
from tof_mra.common import write_cases, write_json, load_config


def summary_rows():
    records = []
    for label, length in ((1, 1.), (2, 9.)):
        record = {"region_label": label, "total_length": length, "volume": length*2,
                  "num_components": 2, "num_bifurcations": 3, "num_loops": 1, "num_abnormal_degree_nodes": 1}
        for root in ROOTS:
            record[root+"mean_curvature"] = 10. if label == 1 else 20.
            record[root+"mean_square_curvature"] = 2. if label == 1 else 4.
        records.append(record)
    return pd.DataFrame(records)


class CaravelTableTests(unittest.TestCase):
    def test_three_roots_and_length_weighted_whole_brain(self):
        frame = normalize_summary(summary_rows(), expected_labels=[1, 2])
        frame["case_id"], frame["model"] = "case001", "baseline"
        whole = aggregate_whole_brain(frame).iloc[0]
        self.assertEqual(whole.total_length, 10.)
        self.assertEqual(whole.num_components, 4.)
        self.assertEqual(whole.root_mean_curvature_avg, 19.)
        self.assertEqual(whole.mean_squared_curvature_avg, 3.8)

    def test_undefined_curvature_does_not_dilute_weighted_mean(self):
        self.assertEqual(weighted_mean([np.nan, 20], [100, 1]), 20)
        self.assertTrue(np.isnan(weighted_mean([1, 2], [0, 0])))

    def test_component_rows_and_corrupt_extensive_values_are_rejected(self):
        frame = pd.concat([summary_rows(), summary_rows()])
        with self.assertRaisesRegex(ValueError, "one region_summary row"):
            normalize_summary(frame, expected_labels=[1, 2])
        frame = summary_rows()
        frame.loc[0, "total_length"] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid extensive"):
            normalize_summary(frame, expected_labels=[1, 2])

    def test_fdr_hand_calculation_and_untestable_family_member(self):
        np.testing.assert_allclose(fdr_bh([.01, .04, .03, .002]), [.02, .04, .04, .008])
        result = fdr_bh([.01, np.nan])
        self.assertAlmostEqual(result[0], .02)
        self.assertTrue(np.isnan(result[1]))

    def test_balance_is_seeded_and_independent_of_input_row_order(self):
        demo = pd.DataFrame({"case_id": [f"c{i:02}" for i in range(15)],
                             "age_years": list(range(10, 25)), "sex": ["M", "M", "F", "F", "M"]*3})
        first, report = balance_sexes(demo)
        second, _ = balance_sexes(demo.sample(frac=1, random_state=8))
        self.assertEqual(first.case_id.tolist(), second.case_id.tolist())
        self.assertTrue((report.balanced_male == report.balanced_female).all())
        self.assertEqual(len(first), 12)

    def test_exact_maximum_age_is_included_in_balance(self):
        demo = pd.DataFrame({"case_id": ["a", "b", "c", "d"], "age_years": [10, 11, 20, 20], "sex": ["M", "F", "M", "F"]})
        balanced, _ = balance_sexes(demo)
        self.assertEqual(len(balanced), 4)

    def test_pooled_t_test_and_male_minus_female_effect(self):
        frame = pd.DataFrame({"sex": ["M"]*6+["F"]*6})
        values = np.r_[np.arange(6)+3, np.arange(6)]
        for feature in FEATURES:
            frame[feature] = values
        result = sex_comparisons(frame).iloc[0]
        self.assertAlmostEqual(result.cohens_d, 3/np.std(np.arange(6), ddof=1))
        self.assertAlmostEqual(result.t_pvalue, ttest_ind(values[:6], values[6:], equal_var=True).pvalue)

    def test_regional_statistics_use_raw_values_and_sample_z_score(self):
        records = []
        for region, offset in ((1, 0.), (2, 10.)):
            for i in range(6):
                records.append({"region_label": region, **{feature: offset+i for feature in FEATURES}})
        tests, means = regional_statistics(pd.DataFrame(records))
        values = np.r_[np.arange(6), np.arange(6)+10]
        expected = (6*25+6*25)/np.square(values-values.mean()).sum()
        self.assertAlmostEqual(tests.iloc[0].eta_squared, expected)
        z = means[means.feature == FEATURES[0]].z_score.to_numpy()
        np.testing.assert_allclose(z, [-1/np.sqrt(2), 1/np.sqrt(2)])

    def test_age_analysis_retains_all_eight_descriptors_with_status(self):
        frame = pd.DataFrame({"age_years": np.arange(19), **{f: np.arange(19)*2 for f in FEATURES}})
        frame[FEATURES[0]] = 1
        result = age_correlations(frame)
        self.assertEqual(len(result), 8)
        self.assertEqual(result.iloc[0].status, "constant_age_or_feature")
        self.assertAlmostEqual(result.iloc[1].pearson_r, 1.)

    def test_metadata_import_requires_explicit_case_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("tof", "label"):
                nib.save(nib.Nifti1Image(np.ones((3,3,3), dtype=np.uint8), np.eye(4)), root/(name+".nii.gz"))
            write_cases(root/"cases.csv", [{"case_id":"test001", "split":"test", "tof":str(root/"tof.nii.gz"), "label":str(root/"label.nii.gz")}])
            pd.DataFrame({"IXI_ID":[123], "AGE":[12.5], "SEX_ID":[2]}).to_csv(root/"source.csv", index=False)
            with self.assertRaisesRegex(ValueError,"explicit"):
                import_demographics(root/"source.csv", root/"cases.csv", root/"output.csv", id_column="IXI_ID")
            pd.DataFrame({"source_id":["123"],"case_id":["test001"]}).to_csv(root/"mapping.csv", index=False)
            import_demographics(root/"source.csv", root/"cases.csv", root/"output.csv", id_column="IXI_ID", case_map=root/"mapping.csv")
            result = pd.read_csv(root/"output.csv")
            self.assertEqual(result.iloc[0].sex,"F")
            self.assertEqual(result.iloc[0].age_years,12.5)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("vedo", "skimage", "networkx", "nipype")), "Optional Caravel extraction dependencies are not installed")
class CaravelExtractionTests(unittest.TestCase):
    def test_graph_features_on_hand_constructed_junction(self):
        import networkx as nx
        from tof_mra._caravel.metrics import compute_metrics_for_mask
        graph = nx.Graph()
        points = {0:[0,0,0],1:[1,0,0],2:[-1,0,0],3:[0,1,0],4:[0,-1,0]}
        for node, pos in points.items():
            graph.add_node(node, pos=np.array(pos,dtype=float))
        for node in range(1,5):
            graph.add_edge(0,node,weight=1.)
        result = compute_metrics_for_mask(graph,{node:1. for node in graph},["total_length","volume","num_bifurcations","num_loops","num_abnormal_degree_nodes"])[0]
        self.assertEqual(result["num_bifurcations"],1)
        self.assertEqual(result["num_abnormal_degree_nodes"],1)
        self.assertEqual(result["num_loops"],0)
        self.assertEqual(result["total_length"],4.)
        self.assertAlmostEqual(result["volume"],4*np.pi)

    def test_real_skeleton_graph_extraction_and_lps_atlas_assignment(self):
        from tof_mra.caravel_extract import extract
        from tof_mra.caravel_data import collect
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            affine=np.diag([.5,.6,.7,1.])
            affine[:3,3]=[40.,50.,60.]
            mask=np.zeros((32,16,16),dtype=np.uint8)
            mask[3:29,8,8]=1
            atlas=np.ones(mask.shape,dtype=np.uint8)
            atlas[16:,:,:]=2
            for name,array in (("tof",mask.astype(np.float32)*500),("label",mask)):
                nib.save(nib.Nifti1Image(array,affine),root/(name+".nii.gz"))
            registered=root/"registered"
            registered.mkdir()
            nib.save(nib.Nifti1Image(atlas,affine),registered/"test001.nii.gz")
            (root/"labels.txt").write_text("1 First\n2 Second\n")
            for model in ("baseline","pediatric"):
                folder=root/"work"/"predictions"/model
                folder.mkdir(parents=True)
                nib.save(nib.Nifti1Image(mask,affine),folder/"test001.nii.gz")
            write_cases(root/"cases.csv",[{"case_id":"test001","split":"test","tof":str(root/"tof.nii.gz"),"label":str(root/"label.nii.gz")}])
            config={"manifest":"cases.csv","work_dir":"work","dataset":{"id":990,"name":"Synthetic","expected_development":0,"expected_test":1},
                    "caravel":{"label_map":"labels.txt","registered_atlases":"registered","region_labels":[1,2]}}
            write_json(root/"study.json",config)
            cfg=load_config(root/"study.json")
            extract(cfg)
            extract(cfg,resume=True)
            regional_path,whole_path=collect(cfg)
            regional=pd.read_csv(regional_path)
            self.assertEqual(set(regional.region_label),{1,2})
            whole=pd.read_csv(whole_path)
            self.assertTrue((whole.total_length>10).all())
            self.assertAlmostEqual(whole.iloc[0].total_length,whole.iloc[1].total_length)


if __name__ == "__main__":
    unittest.main()

