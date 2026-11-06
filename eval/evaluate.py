""" evaluate.py

Development CLI: Dice of the ASAH segmentation model on a folder of CT scans.

Every scan in SOURCE (*.nii, *.nii.gz) is segmented and scored against the
label with the same file name in TARGET. Cases whose label is missing, that
fail to load or segment, or whose label is on a different voxel grid are
skipped with a warning.

Usage (from the repository root):

    python eval/evaluate.py SOURCE TARGET
    python eval/evaluate.py SOURCE TARGET --device cpu --json dice.json

Exit status: 0 if at least one case was scored, 1 otherwise.

"""

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    # Launched as `python eval/evaluate.py`: make `eval` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asah_segmentation import ASAHSegmenter
from eval.metrics import dice_coefficient

DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[1] / "weights" /
                      "asah_nonewnet.pt")
NIFTI_SUFFIXES = (".nii", ".nii.gz")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dice of the ASAH segmentation model on SOURCE scans "
                    "against the same-named labels in TARGET.")
    parser.add_argument("source", type=Path, help="CT scans directory")
    parser.add_argument("target", type=Path, help="reference labels directory")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="model weights (default: weights/"
                             "asah_nonewnet.pt)")
    parser.add_argument("--device", default="auto",
                        help="'auto' (default), 'cpu', 'cuda', ...")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="slices per forward pass (default: 16)")
    parser.add_argument("--label", type=int, default=None,
                        help="label to score (default: every non-zero voxel)")
    parser.add_argument("--json", type=Path, default=None, dest="json_path",
                        metavar="PATH", help="also write the scores to PATH")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    segmenter = ASAHSegmenter.from_checkpoint(
        args.checkpoint, device=args.device, batch_size=args.batch_size)

    scans = sorted(p for p in args.source.iterdir()
                   if p.is_file() and p.name.lower().endswith(NIFTI_SUFFIXES))
    scores: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for scan in scans:
        label = args.target / scan.name
        try:
            if not label.is_file():
                raise FileNotFoundError(f"no label '{label}'")
            mask = segmenter.segment(scan)
            scores[scan.name] = dice_coefficient(mask, label, label=args.label)
        except Exception as exc:
            skipped[scan.name] = f"{type(exc).__name__}: {exc}"
            print(f"warning: skipping {scan.name}: {skipped[scan.name]}",
                  file=sys.stderr, flush=True)
            continue
        print(f"{scan.name}  {scores[scan.name]:.4f}", flush=True)

    values = list(scores.values())
    summary: dict[str, int | float] = {"cases": len(values),
                                       "skipped": len(skipped)}
    if values:
        summary.update(mean=statistics.fmean(values),
                       median=statistics.median(values),
                       min=min(values), max=max(values))
    print("\n" + "  ".join(f"{k} {v:.4f}" if isinstance(v, float)
                           else f"{k} {v}" for k, v in summary.items()))

    if args.json_path is not None:
        report = {"source": str(args.source), "target": str(args.target),
                  "checkpoint": str(args.checkpoint), "label": args.label,
                  **summary, "dice": scores, "skipped_cases": skipped}
        args.json_path.write_text(json.dumps(report, indent=2) + "\n",
                                  encoding="utf-8")
    return 0 if values else 1


if __name__ == "__main__":
    sys.exit(main())
