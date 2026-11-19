""" evaluate.py

Development CLI: agreement between two sets of segmentation masks.

Every mask (*.nii, *.nii.gz) in SOURCE, typically segmented by this package,
is compared with the mask of the same file name in TARGET, the reference:
manual annotations, the legacy TensorFlow model's output, or another model's
masks. Agreement is measured with the Dice coefficient and the absolute volume
difference (AVD, mL). Results are printed per case, then summarised (mean,
standard deviation, median, quartiles, min and max of each metric, and the
worst case for each).

A mask present in only one folder, that fails to load, or whose counterpart
is on a different voxel grid is skipped with a warning.

Usage (from the repository root):

    python eval/evaluate.py SOURCE TARGET
    python eval/evaluate.py SOURCE TARGET --output results.json
    python eval/evaluate.py SOURCE TARGET --label 1 --output results.csv

--output writes a JSON report (per-case results, summary and skipped cases)
or, for a .csv path, one row per case.

Exit status: 0 if at least one case was scored, 1 otherwise.

"""

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

if __package__ in (None, ""):
    # Launched as `python eval/evaluate.py`: make `eval` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import absolute_volume_difference, dice_coefficient

NIFTI_SUFFIXES = (".nii", ".nii.gz")
OUTPUT_SUFFIXES = (".json", ".csv")
# Metric key -> column header.
METRICS = {"dice": "Dice", "avd_ml": "AVD (mL)"}
STATISTICS = ("mean", "std", "median", "q1", "q3", "min", "max")

type Results = dict[str, dict[str, float]]
type Summary = dict[str, dict[str, float | None]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dice and absolute volume difference of the masks in "
                    "SOURCE against the same-named reference masks in TARGET "
                    "(manual annotations, legacy model or another model).")
    parser.add_argument("source", type=Path,
                        help="masks to evaluate, e.g. segmented by this "
                             "package")
    parser.add_argument("target", type=Path,
                        help="reference masks: manual annotations, the legacy "
                             "model's or another model's output")
    parser.add_argument("--label", type=int, default=None,
                        help="label to score (default: every non-zero voxel)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        metavar="PATH",
                        help="also write the results to PATH: a JSON report, "
                             "or per-case rows if PATH ends in .csv")
    args = parser.parse_args(argv)
    for folder in (args.source, args.target):
        if not folder.is_dir():
            parser.error(f"not a directory: '{folder}'")
    if (args.output is not None and
        args.output.suffix.lower() not in OUTPUT_SUFFIXES):
        parser.error("--output must end in .json or .csv")
    return args


def list_masks(folder: Path) -> set[str]:
    """Names of the NIfTI files directly in `folder`."""
    return {p.name for p in folder.iterdir()
            if p.is_file() and p.name.lower().endswith(NIFTI_SUFFIXES)}


def read_mask(path: Path) -> nib.Nifti1Image:
    """Loads a mask into memory once, so that both metrics share it."""
    img = nib.load(path)
    if not isinstance(img, nib.Nifti1Image):  # Nifti2Image subclasses it
        raise TypeError("expected a NIfTI-1 or NIfTI-2 image")
    return nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine)


def score_case(source: Path, target: Path,
               label: int | None) -> dict[str, float]:
    """Dice and AVD (mL) of the mask `source` against the reference `target`."""
    for path, kind in ((source, "source"), (target, "target")):
        if not path.is_file():
            raise FileNotFoundError(f"no {kind} mask '{path}'")
    predicted, reference = read_mask(source), read_mask(target)
    return {"dice": dice_coefficient(predicted, reference, label=label),
            "avd_ml": absolute_volume_difference(predicted, reference,
                                                 label=label)}


def summarize(results: Results) -> Summary:
    """Descriptive statistics of each metric over the scored cases."""
    summary: Summary = {}
    for metric in METRICS:
        values = [r[metric] for r in results.values()]
        if not values:
            continue
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values),
            "q1": q1, "q3": q3, "min": min(values), "max": max(values)}
    return summary


def worst_cases(results: Results) -> dict[str, str]:
    """The case with the lowest Dice and the one with the largest AVD."""
    if not results:
        return {}
    return {"dice": min(results, key=lambda c: results[c]["dice"]),
            "avd_ml": max(results, key=lambda c: results[c]["avd_ml"])}


def _fmt(value: float | None, width: int = 8) -> str:
    return f"{'n/a' if value is None else f'{value:.4f}':>{width}}"


def print_summary(results: Results, skipped: dict[str, str]) -> None:
    print(f"\ncases {len(results)}  skipped {len(skipped)}")
    summary = summarize(results)
    if not summary:
        return
    width = max(map(len, METRICS.values()))
    print("\n" + " " * width +
          "".join(f"{s:>8}" for s in STATISTICS))
    for metric, stats in summary.items():
        print(f"{METRICS[metric]:<{width}}" +
              "".join(_fmt(stats[s]) for s in STATISTICS))
    worst = worst_cases(results)
    print(f"\nlowest Dice:  {worst['dice']} "
          f"({results[worst['dice']]['dice']:.4f})")
    print(f"largest AVD:  {worst['avd_ml']} "
          f"({results[worst['avd_ml']]['avd_ml']:.4f} mL)")


def write_output(path: Path, args: argparse.Namespace, results: Results,
                 skipped: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case", *METRICS, "skipped"])
            for case, r in results.items():
                writer.writerow([case, *(r[m] for m in METRICS), ""])
            for case, reason in skipped.items():
                writer.writerow([case, *("" for _ in METRICS), reason])
        return
    report = {"source": str(args.source), "target": str(args.target),
              "label": args.label, "cases": len(results),
              "skipped": len(skipped), "summary": summarize(results),
              "worst": worst_cases(results), "results": results,
              "skipped_cases": skipped}
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = sorted(list_masks(args.source) | list_masks(args.target))

    results: Results = {}
    skipped: dict[str, str] = {}
    width = max((len(c) for c in cases), default=4)
    print(f"{'case':<{width}}" + "".join(f"{h:>10}" for h in METRICS.values()),
          flush=True)
    for case in cases:
        try:
            results[case] = score_case(args.source / case, args.target / case,
                                       args.label)
        except Exception as exc:
            skipped[case] = f"{type(exc).__name__}: {exc}"
            print(f"warning: skipping {case}: {skipped[case]}",
                  file=sys.stderr, flush=True)
            continue
        print(f"{case:<{width}}" +
              "".join(_fmt(results[case][m], 10) for m in METRICS),
              flush=True)

    print_summary(results, skipped)
    if args.output is not None:
        write_output(args.output, args, results, skipped)
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
