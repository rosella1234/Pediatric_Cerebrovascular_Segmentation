"""Command-line entry points. All orchestration and analyses are Python."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tof-study", description="Pediatric TOF-MRA study pipeline")
    parser.add_argument("--config", type=Path, help="JSON study configuration (paths resolved relative to this file)")
    parser.add_argument("--debug", action="store_true", help="Show exception tracebacks")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Convert explicitly selected TOF/T1 DICOM series using dcm2niix")
    convert.add_argument("--series-manifest", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--dcm2niix", default="dcm2niix")
    convert.add_argument("--decompress-command", help='JSON array, e.g. ["dcmdjpeg"] or ["apptainer","exec","decoder.sif","dcmdjpeg"]')
    manifest = commands.add_parser("make-manifest", help="Build a manifest from one-NIfTI-per-subject folders")
    for option in ("dev-images", "dev-labels", "test-images", "test-labels", "output"):
        manifest.add_argument("--" + option, type=Path, required=True)
    manifest.add_argument("--brain-masks", type=Path)
    inspect = commands.add_parser("inspect", help="Report dimensions, spacing, orientation, image/label geometry")
    inspect.add_argument("--output", type=Path)
    commands.add_parser("prepare", help="Prepare development image/label pairs and separate test inputs")
    for name in ("plan", "train"):
        commands.add_parser(name).add_argument("--dry-run", action="store_true")
    predict = commands.add_parser("predict", help="Independent test inference for one model")
    predict.add_argument("--model", choices=["baseline", "pediatric"], required=True)
    predict.add_argument("--dry-run", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Compute overlap and physical surface distances")
    evaluate.add_argument("--model", choices=["baseline", "pediatric", "both"], default="both")
    evaluate.add_argument("--predictions", type=Path, help="Override prediction folder; requires one model")
    evaluate.add_argument("--output", type=Path, help="Override output CSV; requires one model")
    evaluate.add_argument("--without-brain-mask", action="store_true", help="Explicitly change the source scripts' brain-masked analysis")
    merge = commands.add_parser("merge-reports", help="Join historical overlap/distance CSVs by case ID")
    merge.add_argument("--overlap", type=Path, required=True)
    merge.add_argument("--distance", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Paired Wilcoxon tests, medians/IQRs, and violin plots")
    compare.add_argument("--baseline", type=Path)
    compare.add_argument("--trained", type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--expected-pairs", type=int)
    compare.add_argument("--bahd-normalization", choices=["reference", "prediction"])
    compare.add_argument("--nonfinite-policy", choices=["error", "drop"])
    compare.add_argument("--allow-unpaired", action="store_true", help="Explicitly allow the audited case-set intersection")
    compare.add_argument("--no-plots", action="store_true")
    demo = commands.add_parser("demo", help="Run the evaluator/comparison on synthetic data; no weights or GPU")
    demo.add_argument("--output", type=Path, default=Path("demo-output"))
    caravel_extract = commands.add_parser("caravel-extract", help="Run the supplied Caravel graph and regional metric modules")
    caravel_extract.add_argument("--model", choices=["baseline", "pediatric", "both"], default="both")
    caravel_extract.add_argument("--resume", action="store_true", help="Reuse only completed outputs with matching input hashes")
    commands.add_parser("caravel-collect", help="Collect both models' region_summary.csv files and aggregate whole-brain values")
    analysis = commands.add_parser("caravel-analyze", help="Age/sex/regional analysis and baseline/trained figures")
    analysis.add_argument("--no-plots", action="store_true")
    demographics = commands.add_parser("caravel-demographics", help="Import CSV/XLSX metadata into canonical case_id,age_years,sex CSV")
    demographics.add_argument("--source", type=Path, required=True)
    demographics.add_argument("--output", type=Path, required=True)
    demographics.add_argument("--id-column", default="case_id")
    demographics.add_argument("--case-map", type=Path)
    caravel_demo = commands.add_parser("caravel-demo", help="Exercise the Caravel table/statistics/figure workflow using synthetic summaries")
    caravel_demo.add_argument("--output", type=Path, default=Path("caravel-demo-output"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            from .conversion import convert_series
            decompressor = json.loads(args.decompress_command) if args.decompress_command else None
            if decompressor is not None and not isinstance(decompressor, list):
                raise ValueError("--decompress-command must be a JSON list of tokens.")
            convert_series(args.series_manifest, args.output, dcm2niix=args.dcm2niix, decompressor=decompressor)
            return
        if args.command == "make-manifest":
            from .dataset import make_manifest
            make_manifest(args.dev_images, args.dev_labels, args.test_images, args.test_labels, args.output, args.brain_masks)
            print(args.output.resolve())
            return
        if args.command == "demo":
            from .demo import run_demo
            print(run_demo(args.output))
            return
        if args.command == "caravel-demo":
            from .caravel_demo import run_demo
            print(run_demo(args.output))
            return
        if args.command == "merge-reports":
            from .comparison import merge_reports
            print(merge_reports(args.overlap, args.distance, args.output))
            return
        cfg = load_config(args.config) if args.config else None
        if args.command == "compare":
            from .comparison import compare_reports
            settings = cfg.get("evaluation", {}) if cfg else {}
            work = Path(cfg["work_dir"]) if cfg else None
            baseline = args.baseline or (work / "evaluation" / "baseline.csv" if work else None)
            trained = args.trained or (work / "evaluation" / "pediatric.csv" if work else None)
            output = args.output or (work / "comparison" if work else None)
            if baseline is None or trained is None or output is None:
                raise ValueError("compare needs --config, or all of --baseline, --trained, --output.")
            expected = args.expected_pairs if args.expected_pairs is not None else settings.get("expected_pairs")
            if expected is not None and expected < 1:
                raise ValueError("--expected-pairs must be positive.")
            result = compare_reports(baseline, trained, output,
                bahd_normalization=args.bahd_normalization or settings.get("bahd_normalization", "reference"),
                expected_pairs=expected, allow_unpaired=args.allow_unpaired,
                nonfinite_policy=args.nonfinite_policy or settings.get("nonfinite_policy", "error"),
                font_family=settings.get("font_family", "Arial"), make_plots=not args.no_plots)
            print(result.to_string(index=False))
            return
        if cfg is None:
            raise ValueError(f"{args.command} requires --config PATH before the subcommand.")
        if args.command == "caravel-demographics":
            from .caravel_data import import_demographics
            print(import_demographics(args.source, Path(cfg["manifest"]), args.output, id_column=args.id_column, case_map=args.case_map))
        elif args.command == "caravel-extract":
            from .caravel_extract import extract
            extract(cfg, model=args.model, resume=args.resume)
        elif args.command == "caravel-collect":
            from .caravel_data import collect
            for path in collect(cfg):
                print(path)
        elif args.command == "caravel-analyze":
            from .caravel_stats import analyze
            print(analyze(cfg, make_plots=not args.no_plots))
        elif args.command == "inspect":
            from .dataset import inspect_study
            print(inspect_study(cfg, args.output))
        elif args.command == "prepare":
            from .dataset import prepare_study
            print(prepare_study(cfg))
        elif args.command in {"plan", "train", "predict"}:
            from . import nnunet
            if args.command == "predict":
                nnunet.predict(cfg, args.model, dry_run=args.dry_run)
            else:
                getattr(nnunet, args.command)(cfg, dry_run=args.dry_run)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            if args.model == "both" and (args.output or args.predictions):
                raise ValueError("--output/--predictions overrides require --model baseline or pediatric.")
            if args.without_brain_mask:
                cfg.setdefault("evaluation", {})["brain_mask_policy"] = "none"
            for model in (["baseline", "pediatric"] if args.model == "both" else [args.model]):
                print(evaluate(cfg, model, args.predictions, args.output))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        if args.debug:
            raise
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

