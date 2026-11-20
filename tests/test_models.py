import unittest
from collections import UserDict
from functools import partial
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from PIL import Image

import models


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create({
            "device": "cpu",
            "model": {
                "name": "grounding_dino", "checkpoint": "test", "revision": "test-revision",
                "prompt": "butterfly. moth.", "box_threshold": 0.25, "text_threshold": 0.3,
            },
        })

    def owlv2_config(self):
        return OmegaConf.create({
            "device": "cpu",
            "model": {
                "name": "owlv2", "checkpoint": "test-owlv2", "revision": "test-revision",
                "queries": ["a photo of a butterfly", "a photo of a moth"], "box_threshold": 0.1,
            },
        })

    def owlv2_stub(self, boxes, scores):
        inputs = UserDict({"input_ids": torch.tensor([[101, 102], [103, 104]])})
        inputs.to = Mock(return_value=inputs)
        processor = Mock(return_value=inputs)
        processor.post_process_grounded_object_detection.return_value = [{
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
        }]
        model = Mock(return_value="raw OWLv2 outputs")
        return {"model": model, "processor": processor, "device": torch.device("cpu")}

    def qwen_config(self):
        return OmegaConf.create({
            "device": "cpu",
            "model": {
                "name": "qwen", "checkpoint": "test-qwen", "revision": "test-revision",
                "prompt": "Find adult butterflies and moths; return bbox_2d JSON.",
                "max_new_tokens": 64, "min_pixels": 1024, "max_pixels": 4096,
            },
        })

    def qwen_stub(self, raw):
        inputs = UserDict({"input_ids": torch.tensor([[101, 102]])})
        inputs.to = Mock(return_value=inputs)
        processor = Mock()
        processor.apply_chat_template.return_value = inputs
        processor.batch_decode.return_value = [raw]
        model = Mock()
        model.generate.return_value = torch.tensor([[101, 102, 201, 202, 151645]])
        model.generation_config.eos_token_id = [151645, 151643]
        return {"model": model, "processor": processor, "device": torch.device("cpu")}

    def test_load_preserves_weights_options_device_and_metadata(self):
        processor = Mock()
        processor.image_processor.to_dict.return_value = {"size": {"shortest_edge": 800}}
        model = Mock(dtype=torch.float32)
        model.config._commit_hash = "resolved-revision"
        model.to.return_value = model
        model.eval.return_value = model
        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as processor_loader,
            patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained", return_value=model) as model_loader,
        ):
            loaded = models.load_model(self.cfg)
        processor_loader.assert_called_once_with("test", revision="test-revision")
        model_loader.assert_called_once_with(
            "test", revision="test-revision", dtype=torch.float32,
            use_safetensors=True, disable_custom_kernels=True,
        )
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once_with()
        self.assertIs(loaded["model"], model)
        self.assertIs(loaded["processor"], processor)
        self.assertEqual(loaded["device"], torch.device("cpu"))
        self.assertEqual(loaded["metadata"], {
            "checkpoint": "test", "revision": "resolved-revision", "dtype": "torch.float32",
            "disable_custom_kernels": True, "image_processor": {"size": {"shortest_edge": 800}},
        })

    def test_predict_uses_original_size_and_returns_cpu_lists(self):
        image = Image.new("RGB", (100, 80))
        inputs = UserDict({"input_ids": torch.tensor([[101, 102]])})
        inputs.input_ids = inputs["input_ids"]
        inputs.to = Mock(return_value=inputs)
        processor = Mock(return_value=inputs)
        processor.post_process_grounded_object_detection.return_value = [{
            "boxes": torch.tensor([[10.0, 20.0, 90.0, 60.0]]),
            "scores": torch.tensor([0.75]),
        }]

        def forward(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            return "raw model outputs"

        model = Mock(side_effect=forward)
        loaded = {"model": model, "processor": processor, "device": torch.device("cpu")}
        result = models.predict(image, loaded, self.cfg)
        self.assertEqual(result, ([[10.0, 20.0, 90.0, 60.0]], [0.75]))
        processor.assert_called_once_with(images=image, text="butterfly. moth.", return_tensors="pt")
        inputs.to.assert_called_once_with(torch.device("cpu"))
        model.assert_called_once_with(input_ids=inputs.input_ids)
        processor.post_process_grounded_object_detection.assert_called_once_with(
            "raw model outputs", input_ids=inputs.input_ids, threshold=0.25,
            text_threshold=0.3, target_sizes=[(80, 100)],
        )

    def test_predict_preserves_empty_detection(self):
        inputs = UserDict({"input_ids": torch.tensor([[101, 102]])})
        inputs.input_ids = inputs["input_ids"]
        inputs.to = Mock(return_value=inputs)
        processor = Mock(return_value=inputs)
        processor.post_process_grounded_object_detection.return_value = [{
            "boxes": torch.empty((0, 4)), "scores": torch.empty(0),
        }]
        loaded = {"model": Mock(), "processor": processor, "device": torch.device("cpu")}
        self.assertEqual(models.predict(Image.new("RGB", (100, 80)), loaded, self.cfg), ([], []))

    def test_load_owlv2_uses_eager_and_not_dino_kernel_option(self):
        cfg = self.owlv2_config()
        processor = Mock()
        processor.image_processor.to_dict.return_value = {"do_pad": True, "size": {"height": 960, "width": 960}}
        model = Mock(dtype=torch.float32)
        model.config._commit_hash = "resolved-owlv2-revision"
        model.to.return_value = model
        model.eval.return_value = model
        with (
            patch("models.version", return_value="test-scipy") as package_version,
            patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as processor_loader,
            patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained", return_value=model) as model_loader,
        ):
            loaded = models.load_model(cfg)
        package_version.assert_called_once_with("scipy")
        processor_loader.assert_called_once_with("test-owlv2", revision="test-revision")
        model_loader.assert_called_once_with(
            "test-owlv2", revision="test-revision", dtype=torch.float32,
            use_safetensors=True, attn_implementation="eager",
        )
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once_with()
        self.assertEqual(loaded["metadata"]["revision"], "resolved-owlv2-revision")
        self.assertEqual(loaded["metadata"]["scipy_version"], "test-scipy")
        self.assertTrue(loaded["metadata"]["drop_padding_only_boxes"])
        self.assertNotIn("disable_custom_kernels", loaded["metadata"])

    def test_missing_scipy_fails_before_any_download(self):
        with (
            patch("models.version", side_effect=PackageNotFoundError("scipy")),
            patch("transformers.AutoProcessor.from_pretrained") as processor_loader,
            patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained") as model_loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "OWLv2 requires scipy"):
                models.load_model(self.owlv2_config())
        processor_loader.assert_not_called()
        model_loader.assert_not_called()

    def test_predict_owlv2_uses_two_queries_for_one_image(self):
        cfg = self.owlv2_config()
        image = Image.new("RGB", (100, 80))
        loaded = self.owlv2_stub([[10, 20, 90, 60], [20, 10, 40, 30]], [0.75, 0.5])

        def forward(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            return "raw OWLv2 outputs"

        loaded["model"].side_effect = forward
        boxes, scores = models.predict(image, loaded, cfg)
        self.assertEqual(boxes, [[10, 20, 90, 60], [20, 10, 40, 30]])
        self.assertEqual(scores, [0.75, 0.5])
        queries = [["a photo of a butterfly", "a photo of a moth"]]
        loaded["processor"].assert_called_once_with(images=image, text=queries, return_tensors="pt")
        loaded["model"].assert_called_once_with(input_ids=loaded["processor"].return_value["input_ids"])
        loaded["processor"].post_process_grounded_object_detection.assert_called_once_with(
            "raw OWLv2 outputs", threshold=0.1, target_sizes=[(80, 100)], text_labels=queries,
        )

    def test_predict_owlv2_preserves_empty_and_padding_only_detections(self):
        cfg = self.owlv2_config()
        image = Image.new("RGB", (100, 80))
        for raw_boxes, raw_scores in [([], []), ([[10, 85, 20, 95]], [0.75])]:
            with self.subTest(boxes=raw_boxes):
                self.assertEqual(models.predict(image, self.owlv2_stub(raw_boxes, raw_scores), cfg), ([], []))

    def test_predict_owlv2_filters_padding_without_losing_partial_boxes(self):
        cfg = self.owlv2_config()
        loaded = self.owlv2_stub(
            [[10, 20, 90, 60], [5, 70, 15, 95], [10, 85, 20, 95]], [0.75, 0.5, 0.25],
        )
        self.assertEqual(
            models.predict(Image.new("RGB", (100, 80)), loaded, cfg),
            ([[10, 20, 90, 60], [5, 70, 15, 95]], [0.75, 0.5]),
        )

    def test_predict_owlv2_keeps_invalid_outputs_for_common_validation(self):
        cfg = self.owlv2_config()
        for box, score in [
            ([10, 85, 20, 95], float("nan")),
            ([10, 85, 20, 95], 1.5),
            ([20, 85, 10, 95], 0.75),
            ([float("inf"), 85, float("inf"), 95], 0.75),
        ]:
            with self.subTest(box=box, score=score):
                boxes, scores = models.predict(
                    Image.new("RGB", (100, 80)), self.owlv2_stub([box], [score]), cfg,
                )
                self.assertEqual(len(boxes), 1)
                self.assertEqual(len(scores), 1)

    def test_real_owlv2_postprocessor_handles_landscape_and_portrait_padding(self):
        # Only CPU tensor postprocessing: no pretrained model/tokenizer or SciPy resize.
        from transformers import Owlv2ImageProcessor, Owlv2Processor

        cfg = self.owlv2_config()
        outputs = SimpleNamespace(
            logits=torch.tensor([[[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]]]),
            pred_boxes=torch.tensor([[[0.5, 0.4, 0.2, 0.2], [0.5, 0.9, 0.2, 0.1], [0.9, 0.4, 0.1, 0.2]]]),
        )
        for size, expected in [
            ((100, 80), [[40, 30, 60, 50], [85, 30, 95, 50]]),
            ((80, 100), [[40, 30, 60, 50], [40, 85, 60, 95]]),
        ]:
            with self.subTest(size=size):
                loaded = self.owlv2_stub([], [])
                loaded["model"].return_value = outputs
                processor = loaded["processor"]
                processor.image_processor = Owlv2ImageProcessor()
                processor.post_process_grounded_object_detection.side_effect = partial(
                    Owlv2Processor.post_process_grounded_object_detection, processor,
                )
                boxes, scores = models.predict(Image.new("RGB", size), loaded, cfg)
                torch.testing.assert_close(torch.tensor(boxes), torch.tensor(expected, dtype=torch.float32))
                self.assertEqual(len(scores), 2)

    def test_load_qwen_uses_conditional_generation_fp32_eager_and_pixel_budget(self):
        # Real CPU image processor, but no tokenizer, pretrained weights or model inference.
        from transformers import Qwen2VLImageProcessorFast

        cfg = self.qwen_config()
        processor = Mock(image_processor=Qwen2VLImageProcessorFast(patch_size=16, merge_size=2))
        model = Mock(dtype=torch.float32)
        model.config._commit_hash = "resolved-qwen-revision"
        model.to.return_value = model
        model.eval.return_value = model
        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as processor_loader,
            patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=model) as model_loader,
            patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained") as detector_loader,
            patch("models.version") as package_version,
        ):
            loaded = models.load_model(cfg)
        processor_loader.assert_called_once_with("test-qwen", revision="test-revision")
        model_loader.assert_called_once_with(
            "test-qwen", revision="test-revision", dtype=torch.float32,
            use_safetensors=True, attn_implementation="eager",
        )
        detector_loader.assert_not_called()
        package_version.assert_not_called()
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once_with()
        self.assertEqual(loaded["metadata"]["revision"], "resolved-qwen-revision")
        self.assertEqual(loaded["metadata"]["box_coordinate_system"], "xyxy_0_1000")
        self.assertFalse(loaded["metadata"]["do_sample"])
        self.assertEqual(loaded["metadata"]["image_processor"]["size"], {
            "shortest_edge": 1024, "longest_edge": 4096,
        })
        for size in [(640, 320), (320, 640)]:
            features = processor.image_processor(images=Image.new("RGB", size), return_tensors="pt")
            _, grid_h, grid_w = features["image_grid_thw"][0].tolist()
            self.assertGreaterEqual(grid_h * grid_w * 16 * 16, 1024)
            self.assertLessEqual(grid_h * grid_w * 16 * 16, 4096)

    def test_qwen_generates_without_sampling_trims_prompt_and_maps_original_coordinates(self):
        cfg = self.qwen_config()
        raw = '[{"bbox_2d": [100, 250, 900, 750]}, {"bbox_2d": [0, 0, 1000, 1000]}]'
        image = Image.new("RGB", (200, 80))
        loaded = self.qwen_stub(raw)
        generated = loaded["model"].generate.return_value

        def generate(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            return generated

        loaded["model"].generate.side_effect = generate
        self.assertEqual(models.predict(image, loaded, cfg), (
            [[20, 20, 180, 60], [0, 0, 200, 80]], None,
        ))
        processor = loaded["processor"]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": cfg.model.prompt},
        ]}]
        processor.apply_chat_template.assert_called_once_with(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        inputs = processor.apply_chat_template.return_value
        inputs.to.assert_called_once_with(torch.device("cpu"))
        loaded["model"].generate.assert_called_once_with(
            **inputs, max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None,
        )
        torch.testing.assert_close(processor.batch_decode.call_args.args[0], torch.tensor([[201, 202, 151645]]))
        self.assertEqual(processor.batch_decode.call_args.kwargs, {
            "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
        })
        self.assertEqual(loaded["raw_output"], raw)

    def test_qwen_preserves_empty_detections_and_accepts_one_json_fence(self):
        for raw in ["[]", " \n[]\n", "```json\n[]\n```", "```\n[]\n```"]:
            with self.subTest(raw=raw):
                loaded = self.qwen_stub(raw)
                self.assertEqual(models.predict(Image.new("RGB", (100, 80)), loaded, self.qwen_config()), ([], None))
                self.assertEqual(loaded["raw_output"], raw)

    def test_qwen_malformed_output_is_an_error_and_retains_raw_text(self):
        for raw in [
            "", "No butterflies", "null", "{}", '[{"bbox_2d": [0, 0, 1000, 1000]}',
            'Here are the boxes: []', '[] trailing text', "```json\n[]",
            '[{}]', '[null]', '[[0, 0, 1000, 1000]]',
            '[{"bbox_2d": [0, 0, 10]}]', '[{"bbox_2d": [0, true, 10, 10]}]',
            '[{"bbox_2d": ["0", 0, 10, 10]}]', '[{"bbox_2d": [0, 0, NaN, 10]}]',
            '[{"bbox_2d": [0, 0, Infinity, 10]}]', '[{"bbox_2d": [-1, 0, 10, 10]}]',
            '[{"bbox_2d": [0, 0, 1001, 10]}]', '[{"bbox_2d": [10, 0, 0, 10]}]',
            '[{"bbox_2d": [0, 0, 10, 0]}]',
            '[{"bbox_2d": [0, 0, 10, 10]}, {"bbox_2d": null}]',
        ]:
            with self.subTest(raw=raw):
                loaded = self.qwen_stub(raw)
                with self.assertRaises(ValueError):
                    models.predict(Image.new("RGB", (100, 80)), loaded, self.qwen_config())
                self.assertEqual(loaded["raw_output"], raw)

    def test_qwen_token_limit_without_eos_is_not_a_successful_empty_detection(self):
        cfg = self.qwen_config()
        cfg.model.max_new_tokens = 3
        loaded = self.qwen_stub("[]")
        loaded["model"].generate.return_value = torch.tensor([[101, 102, 201, 202, 203]])
        with self.assertRaisesRegex(ValueError, "max_new_tokens without EOS"):
            models.predict(Image.new("RGB", (100, 80)), loaded, cfg)
        self.assertEqual(loaded["raw_output"], "[]")

    def test_qwen_eos_at_token_limit_is_a_success(self):
        cfg = self.qwen_config()
        cfg.model.max_new_tokens = 3
        for eos in [151645, [151645, 151643]]:
            with self.subTest(eos=eos):
                loaded = self.qwen_stub("[]")
                loaded["model"].generation_config.eos_token_id = eos
                self.assertEqual(models.predict(Image.new("RGB", (100, 80)), loaded, cfg), ([], None))

    def test_qwen_failure_before_decoding_clears_stale_raw_text(self):
        loaded = self.qwen_stub("[]")
        loaded["raw_output"] = "previous image"
        loaded["model"].generate.side_effect = RuntimeError("generation failed")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            models.predict(Image.new("RGB", (100, 80)), loaded, self.qwen_config())
        self.assertNotIn("raw_output", loaded)

    def test_both_functions_reject_unimplemented_model(self):
        self.cfg.model.name = "unknown_model"
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            models.load_model(self.cfg)
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            models.predict(Image.new("RGB", (100, 80)), {}, self.cfg)


if __name__ == "__main__":
    unittest.main()
