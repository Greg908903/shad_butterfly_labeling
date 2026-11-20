import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import predict


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.output = self.root / "output"
        self.cfg = OmegaConf.create({
            "images_dir": str(self.images),
            "output_dir": str(self.output),
            "device": "cpu",
            "save_visualizations": True,
            "model": {
                "name": "grounding_dino", "checkpoint": "test", "revision": "test",
                "prompt": "butterfly. moth.", "box_threshold": 0.25, "text_threshold": 0.25,
            },
        })

    def image(self, filename, size=(100, 80), **kwargs):
        Image.new("RGB", size, "white").save(self.images / filename, **kwargs)

    def records(self):
        return [json.loads(line) for line in (self.output / "predictions.jsonl").read_text().splitlines()]

    def fake_predict(self, image, loaded, cfg):
        if image.width == 100:
            return [[10, 20, 90, 60]], [0.8]
        return [], []

    @contextmanager
    def stub_model(self, inference=None, metadata=None):
        with (
            patch.object(predict, "load_model", return_value={"metadata": metadata or {}}) as loader,
            patch.object(predict, "predict", side_effect=inference or self.fake_predict) as predictor,
        ):
            yield loader, predictor

    def test_hydra_config(self):
        directory = Path(predict.__file__).parent / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
            cfg = compose(config_name="config", overrides=[
                "model=grounding_dino", "device=cpu", "model.box_threshold=0.35",
            ])
        self.assertEqual(cfg.model.name, "grounding_dino")
        self.assertEqual(cfg.model.box_threshold, 0.35)
        self.assertEqual(len(cfg.model.revision), 40)
        self.assertEqual(cfg.device, "cpu")

    def test_owlv2_hydra_config_replaces_dino_settings(self):
        directory = Path(predict.__file__).parent / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
            cfg = compose(config_name="config", overrides=["model=owlv2", "device=cpu"])
        self.assertEqual(cfg.model.name, "owlv2")
        self.assertEqual(cfg.model.checkpoint, "google/owlv2-base-patch16-ensemble")
        self.assertEqual(len(cfg.model.revision), 40)
        self.assertEqual(list(cfg.model.queries), ["a photo of a butterfly", "a photo of a moth"])
        expected = OmegaConf.load(directory / "model/owlv2.yaml")
        self.assertEqual(cfg.model.box_threshold, expected.box_threshold)
        self.assertNotIn("prompt", cfg.model)
        self.assertNotIn("text_threshold", cfg.model)

    def test_owlv2_run_keeps_common_output_contract(self):
        self.image("1.jpg")
        self.image("2.png", size=(60, 40))
        self.cfg.model = {
            "name": "owlv2", "checkpoint": "test-owlv2", "revision": "test",
            "queries": ["a photo of a butterfly", "a photo of a moth"], "box_threshold": 0.1,
        }
        with self.stub_model(metadata={"checkpoint": "test-owlv2"}) as (loader, predictor):
            self.assertEqual(predict.run(self.cfg), 0)
        loader.assert_called_once_with(self.cfg)
        self.assertEqual(predictor.call_count, 2)
        records = self.records()
        self.assertEqual(records[0], {
            "photo_id": 1,
            "boxes": [{"left": 0.1, "top": 0.25, "width": 0.8, "height": 0.5}],
            "error": None,
        })
        self.assertEqual(records[1], {"photo_id": 2, "boxes": [], "error": None})
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual(info["config"]["model"]["name"], "owlv2")
        self.assertEqual(info["status"], "complete")
        self.assertEqual(len(list((self.output / "visualizations").glob("*.jpg"))), 2)

    def qwen_config(self):
        self.cfg.model = OmegaConf.load(Path(predict.__file__).parent / "configs/model/qwen.yaml")
        return self.cfg

    def test_qwen_hydra_config_replaces_detector_settings(self):
        directory = Path(predict.__file__).parent / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
            cfg = compose(config_name="config", overrides=["model=qwen", "device=cpu"])
        self.assertEqual(cfg.model.name, "qwen")
        self.assertEqual(cfg.model.checkpoint, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(cfg.model.revision), 40)
        self.assertIn("moth", cfg.model.prompt)
        self.assertIn("bbox_2d", cfg.model.prompt)
        expected = OmegaConf.load(directory / "model/qwen.yaml")
        for key in ("max_new_tokens", "min_pixels", "max_pixels"):
            self.assertEqual(cfg.model[key], expected[key])
        self.assertLessEqual(cfg.model.min_pixels, cfg.model.max_pixels)
        self.assertNotIn("box_threshold", cfg.model)
        self.assertNotIn("text_threshold", cfg.model)
        self.assertNotIn("queries", cfg.model)

    def test_qwen_run_keeps_output_contract_without_inventing_scores(self):
        self.image("1.jpg")
        self.image("2.png", size=(60, 40))
        self.qwen_config()

        def inference(image, loaded, cfg):
            loaded["raw_output"] = '[{"bbox_2d": [100, 250, 900, 750]}]' if image.width == 100 else "[]"
            boxes, _ = self.fake_predict(image, loaded, cfg)
            return boxes, None

        with self.stub_model(inference=inference) as (loader, predictor):
            self.assertEqual(predict.run(self.cfg), 0)
        loader.assert_called_once_with(self.cfg)
        self.assertEqual(predictor.call_count, 2)
        records = self.records()
        self.assertEqual(records[0], {
            "photo_id": 1, "boxes": [{"left": 0.1, "top": 0.25, "width": 0.8, "height": 0.5}],
            "error": None,
        })
        self.assertEqual(records[1], {"photo_id": 2, "boxes": [], "error": None})
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual(info["status"], "complete")
        self.assertIsNone(info["images"][0]["scores"])
        self.assertIsNone(info["images"][1]["scores"])
        self.assertEqual(info["images"][1]["raw_output"], "[]")
        self.assertEqual(len(list((self.output / "visualizations").glob("*.jpg"))), 2)
        with Image.open(self.output / "visualizations/1.jpg") as result:
            r, g, b = result.getpixel((10, 30))
            self.assertGreater(r, g + 50)

    def test_qwen_error_retains_raw_output_but_does_not_leak_to_unreadable_image(self):
        self.image("1.jpg")
        (self.images / "2.jpg").write_bytes(b"not an image")
        self.qwen_config()

        def inference(image, loaded, cfg):
            loaded["raw_output"] = "incomplete ["
            raise ValueError("malformed model output")

        with self.stub_model(inference=inference):
            with self.assertLogs(predict.log, level="ERROR"):
                self.assertEqual(predict.run(self.cfg), 1)
        self.assertIsNone(self.records()[0]["boxes"])
        self.assertIn("inference:", self.records()[0]["error"])
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual(info["images"][0]["raw_output"], "incomplete [")
        self.assertNotIn("raw_output", info["images"][1])
        self.assertEqual(info["n_errors"], 2)

    def test_invalid_qwen_config_fails_before_loading(self):
        self.image("1.jpg")
        for key, value in [
            ("prompt", " "), ("max_new_tokens", 0), ("max_new_tokens", 1.5),
            ("max_new_tokens", True), ("max_new_tokens", "100"),
            ("min_pixels", 0), ("min_pixels", 1023), ("min_pixels", 2097153),
            ("max_pixels", -1), ("max_pixels", 262143), ("max_pixels", float("nan")),
        ]:
            with self.subTest(key=key, value=value):
                self.qwen_config()
                self.cfg.model[key] = value
                with patch.object(predict, "load_model") as loader:
                    with self.assertRaises(ValueError):
                        predict.run(self.cfg)
                    loader.assert_not_called()

    def test_invalid_owlv2_queries_and_threshold_fail_before_loading(self):
        self.image("1.jpg")
        for queries, threshold in [
            ([], 0.1), ("butterfly", 0.1), ([""], 0.1), ([" "], 0.1),
            ([None], 0.1), (["butterfly", 42], 0.1), (["butterfly"], -0.1),
            (["butterfly"], float("nan")),
        ]:
            with self.subTest(queries=queries, threshold=threshold):
                cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
                cfg.model = {
                    "name": "owlv2", "checkpoint": "test", "revision": "test",
                    "queries": queries, "box_threshold": threshold,
                }
                with patch.object(predict, "load_model") as loader:
                    with self.assertRaises(ValueError):
                        predict.run(cfg)
                    loader.assert_not_called()

    def test_discovery_is_numeric_and_nonrecursive(self):
        for name in ("10.JPG", "2.png", "1.webp"):
            self.image(name)
        (self.images / "README.txt").write_text("ignored")
        nested = self.images / "nested"
        nested.mkdir()
        Image.new("RGB", (10, 10)).save(nested / "3.jpg")
        self.assertEqual([i for i, _ in predict.discover_images(self.images)], [1, 2, 10])

    def test_duplicate_ids_fail(self):
        self.image("123.jpg")
        self.image("00123.png")
        with self.assertRaisesRegex(ValueError, "Duplicate photo_id"):
            predict.discover_images(self.images)

    def test_invalid_ids_fail(self):
        for name in ("butterfly.jpg", "-1.jpg", "١٢٣.jpg"):
            with self.subTest(name=name):
                self.image(name)
                with self.assertRaisesRegex(ValueError, "numeric photo_id"):
                    predict.discover_images(self.images)
                (self.images / name).unlink()

    def test_empty_and_missing_directory_fail(self):
        for directory in (self.images, self.root / "missing"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                predict.discover_images(directory)

    def test_normalize_and_clip(self):
        boxes = predict.normalize_boxes([[10, 20, 90, 60], [-1, -2, 101, 82]], (100, 80))
        self.assertEqual(boxes[0], {"left": 0.1, "top": 0.25, "width": 0.8, "height": 0.5})
        self.assertEqual(boxes[1], {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0})
        self.assertEqual(predict.normalize_boxes([], (100, 80)), [])

    def test_invalid_boxes_fail_instead_of_becoming_empty(self):
        for box in ([10, 20, 10, 60], [90, 20, 10, 60], [101, 1, 110, 2],
                    [float("nan"), 0, 10, 10], [0, 0, float("inf"), 10], [0, 1, 2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                predict.normalize_boxes([box], (100, 80))

    def test_run_writes_detection_empty_and_error(self):
        self.image("123.jpg")
        self.image("124.png", size=(60, 40))
        (self.images / "125.jpg").write_bytes(b"not an image")
        with self.stub_model(metadata={"revision": "test"}) as (loader, predictor):
            with self.assertLogs(predict.log, level="ERROR"):
                self.assertEqual(predict.run(self.cfg), 1)
        loader.assert_called_once_with(self.cfg)
        self.assertEqual(predictor.call_count, 2)
        for call in predictor.call_args_list:
            self.assertIs(call.args[1], loader.return_value)
            self.assertIs(call.args[2], self.cfg)
        records = self.records()
        self.assertEqual([r["photo_id"] for r in records], [123, 124, 125])
        self.assertEqual(len(records[0]["boxes"]), 1)
        self.assertEqual(records[1], {"photo_id": 124, "boxes": [], "error": None})
        self.assertIsNone(records[2]["boxes"])
        self.assertIn("image_load:", records[2]["error"])
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual((info["n_input"], info["n_success"], info["n_errors"]), (3, 2, 1))
        self.assertEqual(info["status"], "incomplete")
        self.assertEqual(info["model"], {"revision": "test"})
        self.assertEqual(info["images"][0]["scores"], [0.8])
        self.assertEqual(len(list((self.output / "visualizations").glob("*.jpg"))), 2)
        with Image.open(self.output / "visualizations" / "123.jpg") as result:
            r, g, b = result.getpixel((10, 30))
            self.assertGreater(r, g + 50)
            self.assertEqual(result.size, (100, 80))

    def test_exif_orientation_and_no_visualizations(self):
        exif = Image.Exif()
        exif[274] = 6
        self.image("1.jpg", size=(80, 100), exif=exif)
        self.cfg.save_visualizations = False
        with self.stub_model():
            self.assertEqual(predict.run(self.cfg), 0)
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual(info["images"][0]["size"], [100, 80])
        self.assertEqual(len(self.records()[0]["boxes"]), 1)
        self.assertFalse((self.output / "visualizations").exists())
        with Image.open(self.images / "1.jpg") as source:
            self.assertEqual(source.size, (80, 100))
            self.assertEqual(source.getexif()[274], 6)

    def test_invalid_model_output_is_an_error(self):
        self.image("1.jpg")
        outputs = [([[1, 1, 1, 2]], [0.8]), ([[0, 0, 10, 10]], []), ([[0, 0, 10, 10]], None),
                   ([[0, 0, 10, 10]], [float("nan")])]
        for index, output in enumerate(outputs):
            with self.subTest(output=output):
                self.cfg.output_dir = str(self.root / f"invalid-{index}")
                self.output = Path(self.cfg.output_dir)
                with self.stub_model(inference=lambda image, loaded, cfg: output):
                    with self.assertLogs(predict.log, level="ERROR"):
                        self.assertEqual(predict.run(self.cfg), 1)
                self.assertIsNone(self.records()[0]["boxes"])
                self.assertIn("inference:", self.records()[0]["error"])

    def test_visualization_error_does_not_erase_prediction(self):
        self.image("1.jpg")
        with self.stub_model():
            with patch.object(predict, "save_visualization", side_effect=OSError("test drawing error")):
                with self.assertLogs(predict.log, level="ERROR"):
                    self.assertEqual(predict.run(self.cfg), 1)
        self.assertIsNone(self.records()[0]["error"])
        self.assertEqual(len(self.records()[0]["boxes"]), 1)
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual(info["n_visualization_errors"], 1)

    def test_does_not_overwrite_results(self):
        self.image("1.jpg")
        self.output.mkdir()
        path = self.output / "predictions.jsonl"
        path.write_text("existing results")
        with patch.object(predict, "load_model") as loader:
            with self.assertRaises(FileExistsError):
                predict.run(self.cfg)
            loader.assert_not_called()
        self.assertEqual(path.read_text(), "existing results")

    def test_model_load_failure_saves_incomplete_run(self):
        self.image("1.jpg")
        with patch.object(predict, "load_model", side_effect=RuntimeError("test load error")):
            with self.assertRaisesRegex(RuntimeError, "test load error"):
                predict.run(self.cfg)
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual(info["status"], "incomplete")
        self.assertEqual(info["n_success"], 0)
        self.assertIn("test load error", info["error"])

    def test_invalid_config_and_missing_cuda_fail_before_model_loading(self):
        self.image("1.jpg")
        for key, value in (("model.box_threshold", -0.1), ("model.text_threshold", float("nan")),
                           ("model.prompt", " "), ("model.name", "unknown_model"), ("device", "mps")):
            with self.subTest(key=key):
                cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
                OmegaConf.update(cfg, key, value)
                with patch.object(predict, "load_model") as loader:
                    with self.assertRaises(ValueError):
                        predict.run(cfg)
                    loader.assert_not_called()
        self.cfg.device = "cuda"
        with patch.object(predict.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "device=cpu"):
                predict.run(self.cfg)

    def test_keyboard_interrupt_preserves_partial_run(self):
        self.image("1.jpg", size=(60, 40))
        self.image("2.jpg")
        calls = 0

        def interrupted(image, loaded, cfg):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            return [], []

        with self.stub_model(inference=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                predict.run(self.cfg)
        self.assertEqual(len(self.records()), 1)
        info = json.loads((self.output / "run.json").read_text())
        self.assertEqual(info["status"], "incomplete")
        self.assertEqual(info["photo_ids"], [1, 2])
        self.assertEqual(info["n_success"], 1)
        self.assertIn("KeyboardInterrupt", info["error"])


if __name__ == "__main__":
    unittest.main()
