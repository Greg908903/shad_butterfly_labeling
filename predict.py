"""Run a configured detector on images named <photo_id>.<extension>."""

import json
import logging
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont, ImageOps

from models import load_model, predict

log = logging.getLogger(__name__)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(directory: Path) -> list[tuple[int, Path]]:
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    images = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if not path.stem.isascii() or not path.stem.isdecimal():
            raise ValueError(f"Expected a numeric photo_id as filename: {path.name}")
        photo_id = int(path.stem)
        if photo_id in images:
            raise ValueError(f"Duplicate photo_id {photo_id}: {images[photo_id].name}, {path.name}")
        images[photo_id] = path
    if not images:
        raise ValueError(f"No JPG, PNG or WebP images in {directory}")
    return sorted(images.items())


def normalize_boxes(boxes: list, size: tuple[int, int]) -> list[dict]:
    """Clip pixel xyxy boxes to the image and convert to normalized xywh."""
    width, height = size
    result = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError(f"Invalid bbox: {box}")
        x1, y1, x2, y2 = box
        x1, x2 = (min(1.0, max(0.0, x / width)) for x in (x1, x2))
        y1, y2 = (min(1.0, max(0.0, y / height)) for y in (y1, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"BBox has no positive area inside the image: {box}")
        result.append({"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1})
    return result


def save_visualization(image: Image.Image, boxes: list[dict], scores: list | None, path: Path):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, min(image.size) // 40))
    line_width = max(2, min(image.size) // 250)
    display_scores = scores if scores is not None else [None] * len(boxes)
    for index, (box, score) in enumerate(zip(boxes, display_scores, strict=True), start=1):
        x1, y1 = box["left"] * image.width, box["top"] * image.height
        x2 = (box["left"] + box["width"]) * image.width
        y2 = (box["top"] + box["height"]) * image.height
        draw.rectangle((x1, y1, x2, y2), outline="red", width=line_width)
        label = str(index) if score is None else f"{index}: {score:.2f}"
        draw.text((x1 + 3, y1 + 3), label, fill="red", font=font)
    canvas.save(path, format="JPEG", quality=95)


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(cfg: DictConfig) -> int:
    started = perf_counter()
    if cfg.model.name == "grounding_dino":
        threshold_keys = ("box_threshold", "text_threshold")
    elif cfg.model.name == "owlv2":
        threshold_keys = ("box_threshold",)
    elif cfg.model.name == "qwen":
        threshold_keys = ()
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}. Use grounding_dino, owlv2 or qwen.")
    for key in threshold_keys:
        if not 0 <= float(cfg.model[key]) <= 1:
            raise ValueError(f"model.{key} must be between 0 and 1.")
    if cfg.model.name in {"grounding_dino", "qwen"}:
        if not isinstance(cfg.model.prompt, str) or not cfg.model.prompt.strip():
            raise ValueError("model.prompt must be a non-empty string.")
    elif cfg.model.name == "owlv2":
        queries = cfg.model.get("queries")
        if not isinstance(queries, (list, ListConfig)) or not queries or any(
            not isinstance(query, str) or not query.strip() for query in queries
        ):
            raise ValueError("model.queries must be a non-empty list of non-empty strings.")
    if cfg.model.name == "qwen":
        for key in ("max_new_tokens", "min_pixels", "max_pixels"):
            value = cfg.model.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"model.{key} must be a positive integer.")
        if not 1024 <= cfg.model.min_pixels <= cfg.model.max_pixels:
            raise ValueError("Qwen requires 1024 <= model.min_pixels <= model.max_pixels.")
    device = torch.device(cfg.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use device=cpu, device=cuda or device=cuda:0.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Use device=cpu or check the NVIDIA driver/PyTorch build.")

    images_dir = Path(to_absolute_path(cfg.images_dir))
    images = discover_images(images_dir)
    output_dir = Path(to_absolute_path(cfg.output_dir))
    for name in ("predictions.jsonl", "run.json", "visualizations"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}; choose another output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_visualizations:
        (output_dir / "visualizations").mkdir()

    info = {
        "status": "incomplete",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "images_dir": str(images_dir),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_num_threads": torch.get_num_threads(),
            "torch_cuda": torch.version.cuda,
            "packages": {name: version(name) for name in (
                "torch", "torchvision", "transformers", "hydra-core", "omegaconf", "Pillow",
                "numpy", "tokenizers", "huggingface-hub", "safetensors",
            )},
        },
        "n_input": len(images),
        "photo_ids": [photo_id for photo_id, _ in images],
        "images": [],
    }
    try:
        log.info("Loading %s on %s", cfg.model.checkpoint, device)
        load_started = perf_counter()
        loaded = load_model(cfg)
        info["model"] = loaded["metadata"]
        synchronize(device)
        info["model_load_sec"] = perf_counter() - load_started

        with (output_dir / "predictions.jsonl").open("x", encoding="utf-8") as destination:
            for index, (photo_id, path) in enumerate(images, start=1):
                # An unreadable image must not inherit the previous Qwen response.
                loaded.pop("raw_output", None)
                record = {"photo_id": photo_id, "boxes": None, "error": None}
                detail = {"photo_id": photo_id, "file": path.name}
                stage = "image_load"
                stage_started = perf_counter()
                try:
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    detail["image_load_sec"] = perf_counter() - stage_started
                    detail["size"] = [image.width, image.height]

                    stage = "inference"
                    synchronize(device)
                    stage_started = perf_counter()
                    boxes, scores = predict(image, loaded, cfg)
                    if scores is None and cfg.model.name != "qwen":
                        raise ValueError("Detector confidence scores must not be None.")
                    if scores is not None and (len(boxes) != len(scores) or not all(
                        math.isfinite(score) and 0 <= score <= 1 for score in scores
                    )):
                        raise ValueError("Invalid scores or mismatched numbers of boxes and scores.")
                    normalized = normalize_boxes(boxes, image.size)
                    synchronize(device)
                    detail["processing_sec"] = perf_counter() - stage_started
                    detail["scores"] = scores
                    record["boxes"] = normalized
                except Exception as exc:
                    record["error"] = f"{stage}: {type(exc).__name__}: {exc}"
                    detail["failed_stage_sec"] = perf_counter() - stage_started
                    log.exception("photo_id=%s failed during %s", photo_id, stage)

                if "raw_output" in loaded:
                    detail["raw_output"] = loaded["raw_output"]
                if record["error"] is None and cfg.save_visualizations:
                    drawing_started = perf_counter()
                    try:
                        save_visualization(
                            image, record["boxes"], scores,
                            output_dir / "visualizations" / f"{photo_id}.jpg",
                        )
                    except Exception as exc:
                        # A drawing failure must not erase successful predictions.
                        detail["visualization_error"] = f"{type(exc).__name__}: {exc}"
                        log.exception("Could not save visualization for photo_id=%s", photo_id)
                    detail["visualization_sec"] = perf_counter() - drawing_started

                detail["error"] = record["error"]
                destination.write(json.dumps(record, allow_nan=False) + "\n")
                destination.flush()
                info["images"].append(detail)
                log.info("[%d/%d] photo_id=%s: %s", index, len(images), photo_id,
                         record["error"] or f"{len(record['boxes'])} boxes")
    except BaseException as exc:
        # Preserve the abort reason (including Ctrl+C), then propagate it.
        info["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        details = info["images"]
        info["n_success"] = sum(item["error"] is None for item in details)
        info["n_errors"] = sum(item["error"] is not None for item in details)
        info["n_visualization_errors"] = sum("visualization_error" in item for item in details)
        info["processing_sec"] = sum(item.get("processing_sec", 0) for item in details)
        info["visualization_sec"] = sum(item.get("visualization_sec", 0) for item in details)
        info["total_sec"] = perf_counter() - started
        if info["n_success"] == len(images) and not info["n_visualization_errors"] and "error" not in info:
            info["status"] = "complete"
        (output_dir / "run.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )

    log.info("Saved results to %s (%s)", output_dir, info["status"])
    return 0 if info["status"] == "complete" else 1


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
