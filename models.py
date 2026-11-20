"""Model loading and inference selected by cfg.model.name."""

from importlib.metadata import PackageNotFoundError, version
import json
import math

import torch
from omegaconf import DictConfig
from PIL import Image


def load_model(cfg: DictConfig) -> dict:
    """Load once; return the model, processor, device and serializable metadata."""
    model_cfg = cfg.model
    metadata = {}
    if model_cfg.name == "grounding_dino":
        model_options = {"disable_custom_kernels": True}
    elif model_cfg.name == "owlv2":
        # The standard OWLv2 image processor uses scipy.ndimage for resizing.
        try:
            metadata["scipy_version"] = version("scipy")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "OWLv2 requires scipy. Update dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        model_options = {"attn_implementation": "eager"}
        metadata["drop_padding_only_boxes"] = True
    elif model_cfg.name == "qwen":
        model_options = {"attn_implementation": "eager"}
        metadata.update({"box_coordinate_system": "xyxy_0_1000", "do_sample": False})
    else:
        raise ValueError(f"Unsupported model: {model_cfg.name}. Use grounding_dino, owlv2 or qwen.")

    device = torch.device(cfg.device)
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_cfg.checkpoint, revision=model_cfg.revision)
    model_class = AutoModelForZeroShotObjectDetection
    if model_cfg.name == "qwen":
        from transformers import Qwen3VLForConditionalGeneration

        model_class = Qwen3VLForConditionalGeneration
        # These size fields are pixel COUNTS, not side lengths (official Qwen processor).
        processor.image_processor.size = {
            "shortest_edge": int(model_cfg.min_pixels),
            "longest_edge": int(model_cfg.max_pixels),
        }
    model = model_class.from_pretrained(
        model_cfg.checkpoint,
        revision=model_cfg.revision,
        dtype=torch.float32,
        use_safetensors=True,
        **model_options,
    ).to(device).eval()

    metadata.update({
        "checkpoint": model_cfg.checkpoint,
        "revision": model.config._commit_hash,
        "dtype": str(model.dtype),
        **model_options,
        "image_processor": processor.image_processor.to_dict(),
    })
    return {
        "model": model,
        "processor": processor,
        "device": device,
        "metadata": metadata,
    }


def predict(image: Image.Image, loaded: dict, cfg: DictConfig) -> tuple[list, list | None]:
    """Return original-image pixel xyxy boxes and scores (None for Qwen); retain its raw_output."""
    model_cfg = cfg.model
    if model_cfg.name == "qwen":
        loaded.pop("raw_output", None)
        processor = loaded["processor"]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": model_cfg.prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(loaded["device"])
        with torch.inference_mode():
            generated = loaded["model"].generate(
                **inputs, max_new_tokens=int(model_cfg.max_new_tokens), do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        generated = generated[:, inputs["input_ids"].shape[-1]:]
        raw = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        loaded["raw_output"] = raw
        eos_ids = loaded["model"].generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        if generated.shape[-1] >= model_cfg.max_new_tokens and generated[0, -1].item() not in eos_ids:
            raise ValueError("Qwen reached model.max_new_tokens without EOS; response may be truncated.")

        # Accept plain JSON or one complete surrounding Markdown fence; never repair/truncate JSON.
        lines = raw.strip().splitlines()
        if len(lines) >= 3 and lines[0] in {"```json", "```"} and lines[-1] == "```":
            raw = "\n".join(lines[1:-1])
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("Qwen response must be a JSON array of objects with bbox_2d.")
        boxes = []
        for item in items:
            box = item.get("bbox_2d") if isinstance(item, dict) else None
            if not isinstance(box, list) or len(box) != 4 or not all(
                type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1000
                for value in box
            ):
                raise ValueError(f"Invalid Qwen bbox_2d (expected four numbers in [0, 1000]): {item}")
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Qwen bbox_2d has no positive area: {box}")
            boxes.append([
                x1 / 1000 * image.width, y1 / 1000 * image.height,
                x2 / 1000 * image.width, y2 / 1000 * image.height,
            ])
        return boxes, None

    if model_cfg.name == "grounding_dino":
        text = model_cfg.prompt
    elif model_cfg.name == "owlv2":
        # Two text queries for one image, not a batch of two images.
        text = [list(model_cfg.queries)]
    else:
        raise ValueError(f"Unsupported model: {model_cfg.name}. Use grounding_dino, owlv2 or qwen.")

    processor = loaded["processor"]
    inputs = processor(images=image, text=text, return_tensors="pt").to(loaded["device"])
    with torch.inference_mode():
        outputs = loaded["model"](**inputs)
    if model_cfg.name == "grounding_dino":
        result = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=float(model_cfg.box_threshold),
            text_threshold=float(model_cfg.text_threshold),
            target_sizes=[(image.height, image.width)],
        )[0]
    else:
        # OWLv2's processor rescales from the padded square to original-image pixels.
        result = processor.post_process_grounded_object_detection(
            outputs,
            threshold=float(model_cfg.box_threshold),
            target_sizes=[(image.height, image.width)],
            text_labels=text,
        )[0]

    boxes, scores = result["boxes"], result["scores"]
    if model_cfg.name == "owlv2":
        x1, y1, x2, y2 = boxes.unbind(-1)
        outside = (x1 >= image.width) | (y1 >= image.height) | (x2 <= 0) | (y2 <= 0)
        # Drop valid boxes wholly in synthetic padding. Keep invalid outputs so
        # the common validator reports an error rather than silently hiding it.
        invalid = (
            ~torch.isfinite(boxes).all(-1) | (x2 <= x1) | (y2 <= y1)
            | ~torch.isfinite(scores) | (scores < 0) | (scores > 1)
        )
        keep = ~outside | invalid
        boxes, scores = boxes[keep], scores[keep]
    return boxes.cpu().tolist(), scores.cpu().tolist()
