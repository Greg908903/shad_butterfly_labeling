"""Evaluate saved boxes on all reference IDs; no models or images are loaded."""

import csv
import hashlib
import json
import logging
import math
import platform
import sys
from pathlib import Path
from statistics import median

import hydra
import numpy as np
import scipy
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

log = logging.getLogger(__name__)
THRESHOLDS = (0.5, 0.75)


def read_records(path: Path) -> dict:
    """Accept a JSON array (or single record) and JSONL, with the same record schema."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        records = json.loads(text)
        if isinstance(records, dict):
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty JSON array or JSONL of photo_id/boxes records.")
    indexed = {}
    for record in records:
        photo_id = record.get("photo_id") if isinstance(record, dict) else None
        if type(photo_id) is not int or photo_id < 0:
            raise ValueError(f"{path}: photo_id must be a non-negative integer: {record}")
        if photo_id in indexed:
            raise ValueError(f"{path}: duplicate photo_id={photo_id}")
        indexed[photo_id] = record
    return indexed


def xyxy(record: dict) -> list:
    """Validate normalized xywh. Tolerate only 1e-7 coordinate rounding at the edge."""
    error = record.get("error")
    if error is not None:
        raise ValueError(f"Recorded processing error: {error}")
    if not isinstance(record.get("boxes"), list):
        raise ValueError("boxes must be a list; [] is a valid negative answer, null is not.")
    result = []
    for box in record["boxes"]:
        if not isinstance(box, dict):
            raise ValueError(f"Expected a normalized xywh object: {box}")
        values = [box.get(key) for key in ("left", "top", "width", "height")]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
            raise ValueError(f"Invalid bbox numbers: {box}")
        left, top, width, height = values
        if not (0 <= left < 1 and 0 <= top < 1 and width > 0 and height > 0
                and left + width <= 1 + 1e-7 and top + height <= 1 + 1e-7):
            raise ValueError(f"BBox must have positive area inside [0, 1]: {box}")
        result.append([left, top, min(1.0, left + width), min(1.0, top + height)])
    return result


def iou_matrix(reference: list, predictions: list) -> np.ndarray:
    a = np.asarray(reference, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    intersection = np.maximum(
        0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]),
    ).prod(axis=-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def match_boxes(ious: np.ndarray, threshold: float) -> dict:
    n_reference, n_predictions = ious.shape
    # Construct from a dense boolean array: false values must NOT be stored as sparse edges.
    matching = maximum_bipartite_matching(csr_matrix(ious >= threshold), perm_type="column")
    pairs = [
        {"reference_index": i, "prediction_index": int(j), "iou": float(ious[i, j])}
        for i, j in enumerate(matching) if j >= 0
    ]
    return {"tp": len(pairs), "fp": n_predictions - len(pairs),
            "fn": n_reference - len(pairs), "matches": pairs}


def summarize(rows: list) -> dict:
    """Micro metrics, never the mean of per-image F1. Undefined ratios are null."""
    result = {
        "n_images": len(rows), "n_reference_boxes": sum(row["n_reference_boxes"] for row in rows),
        "n_errors": sum(row["error"] is not None for row in rows),
        "metrics": None, "negative_images": None,
    }
    if result["n_errors"]:
        # Do not publish successful-only metrics as if they covered the whole selected set.
        return result
    result["metrics"] = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        tp, fp, fn = (sum(row["metrics"][key][name] for row in rows) for name in ("tp", "fp", "fn"))
        result["metrics"][key] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "n_detection_error_images": sum(
                row["metrics"][key]["fp"] + row["metrics"][key]["fn"] > 0 for row in rows
            ),
        }
    negatives = [row for row in rows if row["n_reference_boxes"] == 0]
    n_false_positive_images = sum(row["n_prediction_boxes"] > 0 for row in negatives)
    result["negative_images"] = {
        "n_images": len(negatives),
        "false_positive_boxes": sum(row["n_prediction_boxes"] for row in negatives),
        "images_with_false_positives": n_false_positive_images,
        "image_fp_rate": n_false_positive_images / len(negatives) if negatives else None,
    }
    return result


def file_info(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_leaderboard_table(rows: list, path: Path, reference_path: Path):
    """Render the same micro metrics as leaderboard.csv, without a GUI or recomputation."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError(
            "The PNG table requires matplotlib. Run python -m pip install -r requirements.txt "
            "or use save_table=false. JSON/CSV metrics have already been saved."
        ) from exc

    metrics = ("tp", "fp", "fn", "precision", "recall", "f1")
    labels = ["Model", "Errors"] + [
        f"{metric.upper() if metric in {'tp', 'fp', 'fn', 'f1'} else metric.title()}\n@{threshold:g}"
        for threshold in THRESHOLDS for metric in metrics
    ]
    cells = []
    for row in rows:
        values = [row["run"], str(row["n_errors"])]
        for threshold in THRESHOLDS:
            for metric in metrics:
                value = row.get(f"{metric}@{threshold}")
                values.append("--" if value is None else (
                    str(value) if metric in {"tp", "fp", "fn"} else f"{value:.4f}"
                ))
        cells.append(values)

    figure = Figure(figsize=(18, 2.1 + 0.46 * len(rows)), facecolor="white")
    FigureCanvasAgg(figure)
    try:
        figure.text(0.02, 0.94, f"Detection metrics (micro) | {rows[0]['n_images']} images",
                    fontsize=15, fontweight="bold", va="top")
        figure.text(0.02, 0.83, f"Reference: {reference_path.parent.name}/{reference_path.name}",
                    fontsize=9, color="#526170")
        axes = figure.add_axes([0.02, 0.23, 0.96, 0.52])
        axes.set_axis_off()
        table = axes.table(cellText=cells, colLabels=labels, cellLoc="center",
                           colWidths=[0.18, 0.055] + [0.765 / 12] * 12, bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#dce3ea")
            if row == 0:
                cell.set_facecolor("#243447" if column < 2 else (
                    "#214c74" if column < 8 else "#386078"
                ))
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("white" if row % 2 else "#f3f6fa")
                if column in {7, 13}:
                    cell.set_text_props(weight="bold")
                if column == 1 and rows[row - 1]["n_errors"]:
                    cell.set_text_props(color="#b42318", weight="bold")
        figure.text(0.02, 0.08,
                    "@ = IoU threshold. Precision / Recall / F1: 0-1. Errors: technical processing errors.\n"
                    "-- = undefined or unavailable (not zero); metrics are withheld for incomplete runs.",
                    fontsize=9, color="#526170", va="center")
        with path.open("xb") as destination:
            figure.savefig(destination, format="png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    finally:
        figure.clear()


def read_timing(path: Path, predictions_path: Path, predictions: dict) -> tuple[dict, dict]:
    """Bind timing to a sibling run file and its complete prediction-ID set, not row order."""
    if path.parent.resolve() != predictions_path.parent.resolve():
        raise ValueError("run_path must be in the same folder as predictions_path (metadata from the same run).")
    info = json.loads(path.read_text(encoding="utf-8"))
    details = {}
    for row in info.get("images", []):
        photo_id = row.get("photo_id")
        if type(photo_id) is not int or photo_id in details:
            raise ValueError(f"{path}: invalid or duplicate timing photo_id={photo_id}")
        details[photo_id] = row
    if set(details) != set(predictions):
        raise ValueError(f"{path}: per-image IDs do not match the prediction file.")
    for photo_id, prediction in predictions.items():
        detail = details[photo_id]
        if detail.get("error") != prediction.get("error"):
            raise ValueError(f"{path}: error state does not match predictions for photo_id={photo_id}")
        scores = detail.get("scores")
        if prediction.get("error") is None and scores is not None and len(scores) != len(prediction.get("boxes") or []):
            raise ValueError(f"{path}: box/score counts do not match for photo_id={photo_id}")
    return info, details


def run(cfg: DictConfig) -> int:
    reference_path = Path(to_absolute_path(cfg.reference_path))
    reference_records = read_records(reference_path)
    references = {photo_id: xyxy(record) for photo_id, record in reference_records.items()}
    if cfg.scope not in {"reference", "exact"}:
        raise ValueError("scope must be reference (all reference IDs) or exact (identical ID sets).")
    if not cfg.runs:
        raise ValueError("runs must contain at least one named prediction file.")

    groups = {}
    metadata_path = Path(to_absolute_path(cfg.metadata_path)) if cfg.get("metadata_path") else None
    if metadata_path:
        with metadata_path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                photo_id = int(row["photo_id"])
                if photo_id in groups or not row["source_group"]:
                    raise ValueError(f"Invalid/duplicate metadata for photo_id={photo_id}")
                groups[photo_id] = row["source_group"]
        if references.keys() - groups.keys():
            raise ValueError("Metadata must cover every reference photo_id.")

    output_dir = Path(to_absolute_path(cfg.output_dir))
    for name in ("metrics.json", "leaderboard.csv", "per_image.jsonl", "leaderboard.png"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}; choose another output_dir.")
    report = {
        "reference": {**file_info(reference_path),
                      "n_images": len(references), "n_boxes": sum(map(len, references.values())),
                      "n_positive_images": sum(bool(boxes) for boxes in references.values()),
                      "n_negative_images": sum(not boxes for boxes in references.values())},
        "config": OmegaConf.to_container(cfg, resolve=True),
        "scope": cfg.scope, "evaluated_photo_ids": sorted(references),
        "matching": "maximum_cardinality_one_to_one; IoU >= threshold; micro aggregation",
        "undefined_ratios": "null", "coordinate_rounding_tolerance": 1e-7,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "metadata": file_info(metadata_path) if metadata_path else None,
        "runs": {},
    }
    all_rows, leaderboard = [], []
    for name, options in cfg.runs.items():
        predictions_path = Path(to_absolute_path(options.predictions_path))
        predictions = read_records(predictions_path)
        missing = references.keys() - predictions.keys()
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} reference IDs: {sorted(missing)}")
        extra = predictions.keys() - references.keys()
        if cfg.scope == "exact" and extra:
            raise ValueError(f"{name}: scope=exact forbids {len(extra)} extra prediction IDs.")
        run_path = Path(to_absolute_path(options.run_path)) if options.get("run_path") else None
        run_info, details = read_timing(run_path, predictions_path, predictions) if run_path else ({}, {})
        rows = []
        for photo_id in sorted(references):
            row = {"run": name, "photo_id": photo_id, "source_group": groups.get(photo_id),
                   "n_reference_boxes": len(references[photo_id]), "n_prediction_boxes": None,
                   "error": None, "metrics": None, "processing_sec": None}
            try:
                boxes = xyxy(predictions[photo_id])
                row["n_prediction_boxes"] = len(boxes)
                ious = iou_matrix(references[photo_id], boxes)
                row["metrics"] = {str(t): match_boxes(ious, t) for t in THRESHOLDS}
            except ValueError as exc:
                row["error"] = str(exc)
            if run_path and row["error"] is None:
                seconds = details[photo_id].get("processing_sec")
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError(f"{run_path}: invalid/missing processing_sec for photo_id={photo_id}")
                row["processing_sec"] = seconds
            rows.append(row)
        result = summarize(rows)
        result.update({
            "status": "complete" if not result["n_errors"] else "incomplete",
            "predictions": file_info(predictions_path), "n_prediction_records": len(predictions),
            "n_ignored_prediction_records": len(extra),
            "groups": {group: summarize([row for row in rows if row["source_group"] == group])
                       for group in sorted({groups[i] for i in references})} if groups else {},
            "model_config": run_info.get("config", {}).get("model"),
            "inference_environment": run_info.get("environment"), "timing": None,
        })
        if run_path:
            times = [row["processing_sec"] for row in rows if row["processing_sec"] is not None]
            result["timing"] = {
                **file_info(run_path), "n_timed_images": len(times),
                "processing_sec": sum(times) if len(times) == len(rows) else None,
                "mean_processing_sec": sum(times) / len(times) if len(times) == len(rows) else None,
                "median_processing_sec": median(times) if len(times) == len(rows) else None,
                "scope": "sum of selected image processing times within the original run; not standalone subset wall time",
                "full_run": {key: run_info.get(key) for key in (
                    "status", "n_input", "n_success", "n_errors", "model_load_sec", "processing_sec", "total_sec",
                )},
            }
        report["runs"][name] = result
        all_rows.extend(rows)
        flat = {"run": name, "n_images": len(rows),
                "status": result["status"], "n_errors": result["n_errors"],
                "n_ignored_predictions": len(extra)}
        for threshold in THRESHOLDS:
            metrics = (result["metrics"] or {}).get(str(threshold), {})
            flat.update({f"{key}@{threshold}": metrics.get(key) for key in (
                "tp", "fp", "fn", "precision", "recall", "f1", "n_detection_error_images",
            )})
        flat.update({f"negative_{key}": (result["negative_images"] or {}).get(key) for key in (
            "n_images", "false_positive_boxes", "images_with_false_positives", "image_fp_rate",
        )})
        flat.update({key: (result["timing"] or {}).get(key) for key in ("processing_sec", "mean_processing_sec")})
        leaderboard.append(flat)
        log.info("%s: %d reference images, %d ignored predictions, F1@0.5=%s, F1@0.75=%s, errors=%d",
                 name, len(rows), len(extra), flat["f1@0.5"], flat["f1@0.75"], result["n_errors"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    with (output_dir / "leaderboard.csv").open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(leaderboard[0]))
        writer.writeheader()
        writer.writerows(leaderboard)
    with (output_dir / "per_image.jsonl").open("x", encoding="utf-8") as destination:
        for row in all_rows:
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if cfg.get("save_table", True):
        save_leaderboard_table(leaderboard, output_dir / "leaderboard.png", reference_path)
    log.info("Evaluation uses exactly %d reference IDs. Saved to %s", len(references), output_dir)
    return int(any(result["n_errors"] for result in report["runs"].values()))


@hydra.main(version_base="1.3", config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
