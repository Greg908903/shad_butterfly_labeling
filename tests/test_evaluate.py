import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import evaluate


BOX = {"left": 0.1, "top": 0.1, "width": 0.2, "height": 0.2}


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.run_path = self.root / "run.json"
        self.output = self.root / "evaluation"
        self.metadata = self.root / "metadata.csv"
        self.metadata.write_text("photo_id,source_group\n1,moths\n2,moths\n3,plantae\n4,plantae\n")
        self.reference_rows = [
            {"photo_id": 1, "boxes": [BOX]}, {"photo_id": 2, "boxes": [BOX]},
            {"photo_id": 3, "boxes": []},
        ]
        self.prediction_rows = [
            {"photo_id": 1, "boxes": [BOX, BOX], "error": None},
            {"photo_id": 2, "boxes": [], "error": None},
            {"photo_id": 3, "boxes": [BOX], "error": None},
            {"photo_id": 4, "boxes": None, "error": "unselected image failed"},
        ]
        self.write_jsonl(self.reference, self.reference_rows)
        self.write_jsonl(self.predictions, self.prediction_rows)
        self.timing = {
            "status": "incomplete", "n_input": 4, "n_success": 3, "n_errors": 1,
            "total_sec": 100, "processing_sec": 6,
            "config": {"model": {"name": "test"}}, "environment": {"device_name": "test GPU"},
            # Different order intentionally: joining by row position is incorrect.
            "images": [
                {"photo_id": 3, "processing_sec": 3, "scores": [0.8], "error": None},
                {"photo_id": 1, "processing_sec": 1, "scores": [0.8, 0.8], "error": None},
                {"photo_id": 2, "processing_sec": 2, "scores": [], "error": None},
                {"photo_id": 4, "error": "unselected image failed"},
            ],
        }
        self.run_path.write_text(json.dumps(self.timing))
        self.cfg = OmegaConf.create({
            "reference_path": str(self.reference),
            "scope": "reference", "metadata_path": str(self.metadata),
            "runs": {"test": {"predictions_path": str(self.predictions), "run_path": str(self.run_path)}},
            "save_table": False,
            "output_dir": str(self.output),
        })

    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def report(self):
        return json.loads((self.output / "metrics.json").read_text())

    def test_hydra_config_has_explicit_files_and_reference_scope(self):
        directory = Path(evaluate.__file__).parent / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
            cfg = compose(config_name="evaluate", overrides=["runs.qwen.run_path=null"])
        self.assertEqual(cfg.scope, "reference")
        self.assertEqual(set(cfg.runs), {"grounding_dino", "owlv2", "qwen"})
        self.assertIsNone(cfg.runs.qwen.run_path)
        self.assertTrue(cfg.save_table)

    def test_table_is_enabled_by_default_and_receives_the_same_rows_as_csv(self):
        del self.cfg["save_table"]
        with patch.object(evaluate, "save_leaderboard_table") as renderer:
            self.assertEqual(evaluate.run(self.cfg), 0)
        renderer.assert_called_once()
        rows, path, reference_path = renderer.call_args.args
        self.assertEqual(path, self.output / "leaderboard.png")
        self.assertEqual(reference_path, self.reference)
        with (self.output / "leaderboard.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(csv_rows))
        for row, csv_row in zip(rows, csv_rows):
            self.assertEqual({key: "" if value is None else str(value) for key, value in row.items()}, csv_row)

    def test_table_can_be_disabled_without_importing_matplotlib(self):
        with patch.object(evaluate, "save_leaderboard_table") as renderer:
            self.assertEqual(evaluate.run(self.cfg), 0)
        renderer.assert_not_called()
        self.assertFalse((self.output / "leaderboard.png").exists())
        self.assertTrue((self.output / "metrics.json").is_file())

    def test_table_shows_both_thresholds_and_does_not_turn_missing_values_into_zero(self):
        row = {"run": "qwen", "n_images": 100, "n_errors": 0}
        for threshold, counts, values in [
            (0.5, (53, 0, 1), (1.0, 53 / 54, 106 / 107)),
            (0.75, (52, 1, 2), (52 / 53, 52 / 54, 104 / 107)),
        ]:
            row.update({f"{metric}@{threshold}": value for metric, value in zip(
                ("tp", "fp", "fn", "precision", "recall", "f1"), (*counts, *values),
            )})
        empty = {"run": "empty", "n_images": 100, "n_errors": 0}
        for threshold in evaluate.THRESHOLDS:
            empty.update({f"{metric}@{threshold}": 0 for metric in ("tp", "fp", "fn")})
        failed = {"run": "failed", "n_images": 100, "n_errors": 1}
        figure = Mock()
        axes = figure.add_axes.return_value
        header, f1, error = Mock(), Mock(), Mock()
        axes.table.return_value.get_celld.return_value = {(0, 2): header, (1, 7): f1, (3, 1): error}
        figure_factory, canvas = Mock(return_value=figure), Mock()
        modules = {name: ModuleType(name) for name in (
            "matplotlib", "matplotlib.backends", "matplotlib.backends.backend_agg", "matplotlib.figure",
        )}
        modules["matplotlib.backends.backend_agg"].FigureCanvasAgg = canvas
        modules["matplotlib.figure"].Figure = figure_factory
        path = self.root / "table.png"
        with patch.dict(sys.modules, modules):
            evaluate.save_leaderboard_table([row, empty, failed], path, self.reference)
        canvas.assert_called_once_with(figure)
        options = axes.table.call_args.kwargs
        self.assertEqual(options["colLabels"], ["Model", "Errors"] + [
            f"{label}\n@{threshold:g}" for threshold in evaluate.THRESHOLDS
            for label in ("TP", "FP", "FN", "Precision", "Recall", "F1")
        ])
        self.assertEqual(options["cellText"][0], [
            "qwen", "0", "53", "0", "1", "1.0000", "0.9815", "0.9907",
            "52", "1", "2", "0.9811", "0.9630", "0.9720",
        ])
        self.assertEqual(options["cellText"][1], ["empty", "0"] + ["0", "0", "0", "--", "--", "--"] * 2)
        self.assertEqual(options["cellText"][2], ["failed", "1"] + ["--"] * 12)
        self.assertAlmostEqual(sum(options["colWidths"]), 1)
        error.set_text_props.assert_called_once_with(color="#b42318", weight="bold")
        figure.savefig.assert_called_once()
        self.assertEqual(figure.savefig.call_args.kwargs["format"], "png")
        figure.clear.assert_called_once_with()

    def test_missing_matplotlib_has_an_actionable_message_and_keeps_numeric_reports(self):
        self.cfg.save_table = True
        with patch.dict(sys.modules, {"matplotlib.backends.backend_agg": None}):
            with self.assertRaisesRegex(RuntimeError, "save_table=false"):
                evaluate.run(self.cfg)
        for name in ("metrics.json", "leaderboard.csv", "per_image.jsonl"):
            self.assertTrue((self.output / name).is_file())
        self.assertFalse((self.output / "leaderboard.png").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_real_matplotlib_saves_a_readable_png_without_gui(self):
        from PIL import Image

        self.cfg.save_table = True
        self.assertEqual(evaluate.run(self.cfg), 0)
        with Image.open(self.output / "leaderboard.png") as image:
            image.load()
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 1000)
            self.assertGreater(image.height, 100)

    def test_does_not_overwrite_an_existing_png(self):
        self.output.mkdir()
        path = self.output / "leaderboard.png"
        path.write_bytes(b"existing PNG")
        with self.assertRaises(FileExistsError):
            evaluate.run(self.cfg)
        self.assertEqual(path.read_bytes(), b"existing PNG")
        self.assertFalse((self.output / "metrics.json").exists())

    def test_reads_json_array_single_record_and_jsonl(self):
        for text in [json.dumps(self.reference_rows), "\n".join(map(json.dumps, self.reference_rows))]:
            self.reference.write_text(text)
            self.assertEqual(set(evaluate.read_records(self.reference)), {1, 2, 3})
        self.reference.write_text(json.dumps(self.reference_rows[0], indent=2))
        self.assertEqual(set(evaluate.read_records(self.reference)), {1})

    def test_bad_ids_duplicate_records_and_unsupported_json_fail(self):
        for rows in [[], [{"photo_id": True, "boxes": []}], [{"photo_id": "1", "boxes": []}],
                     [{"photo_id": -1, "boxes": []}], self.reference_rows * 2,
                     {"annotations": self.reference_rows}]:
            with self.subTest(rows=rows):
                self.reference.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    evaluate.read_records(self.reference)

    def test_box_validation_and_only_tiny_edge_rounding_is_allowed(self):
        for box in [
            {**BOX, "width": 0}, {**BOX, "height": -1}, {**BOX, "left": -0.001},
            {**BOX, "width": 1}, {**BOX, "left": float("nan")}, {**BOX, "height": True},
            {**BOX, "width": "0.2"}, {}, [0, 0, 1, 1],
        ]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                evaluate.xyxy({"boxes": [box]})
        self.assertEqual(evaluate.xyxy({"boxes": []}), [])
        rounded = {"left": 0.8, "top": 0, "width": 0.20000001, "height": 1}
        self.assertEqual(evaluate.xyxy({"boxes": [rounded]}), [[0.8, 0, 1.0, 1.0]])
        for record in [{"boxes": None}, {"boxes": [], "error": "failed"}]:
            with self.assertRaises(ValueError):
                evaluate.xyxy(record)

    def test_iou_geometry_and_empty_inputs(self):
        boxes = [[0, 0, 1, 1], [0, 0, 0.5, 1], [0.5, 0, 1, 1]]
        matrix = evaluate.iou_matrix(boxes[:1], boxes)
        np.testing.assert_allclose(matrix, [[1, 0.5, 0.5]])
        self.assertEqual(evaluate.iou_matrix([], boxes).shape, (0, 3))
        self.assertEqual(evaluate.iou_matrix(boxes, []).shape, (3, 0))
        self.assertEqual(evaluate.iou_matrix([], []).shape, (0, 0))

    def test_matching_is_maximum_cardinality_not_greedy_or_maximum_iou_sum(self):
        # A greedy 0.9 match leaves only one pair; the correct 0.6+0.6 matching has two.
        ious = np.array([[0.9, 0.6], [0.6, 0.4]])
        result = evaluate.match_boxes(ious, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))
        self.assertEqual(evaluate.match_boxes(ious, 0.75)["tp"], 1)

    def test_threshold_is_inclusive_and_false_sparse_edges_are_absent(self):
        self.assertEqual(evaluate.match_boxes(np.array([[0.5]]), 0.5)["tp"], 1)
        self.assertEqual(evaluate.match_boxes(np.array([[np.nextafter(0.5, 0)]]), 0.5)["tp"], 0)
        self.assertEqual(evaluate.match_boxes(np.zeros((2, 3)), 0.5)["tp"], 0)

    def test_duplicates_and_empty_matchings(self):
        for shape, expected in [((0, 0), (0, 0, 0)), ((0, 2), (0, 2, 0)),
                                ((2, 0), (0, 0, 2)), ((1, 2), (1, 1, 0))]:
            with self.subTest(shape=shape):
                result = evaluate.match_boxes(np.ones(shape), 0.5)
                self.assertEqual(tuple(result[k] for k in ("tp", "fp", "fn")), expected)

    def test_run_scores_all_reference_ids_ignores_extras_and_sums_selected_times(self):
        self.assertEqual(evaluate.run(self.cfg), 0)
        report = self.report()
        self.assertEqual(report["evaluated_photo_ids"], [1, 2, 3])
        self.assertEqual(report["reference"]["n_images"], 3)
        result = report["runs"]["test"]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["n_ignored_prediction_records"], 1)
        self.assertEqual(result["n_errors"], 0)
        for metrics in result["metrics"].values():
            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 2, 1))
            self.assertAlmostEqual(metrics["precision"], 1 / 3)
            self.assertEqual(metrics["recall"], 0.5)
            self.assertEqual(metrics["f1"], 0.4)  # micro, NOT mean image F1
            self.assertEqual(metrics["n_detection_error_images"], 3)
        self.assertEqual(result["negative_images"], {
            "n_images": 1, "false_positive_boxes": 1, "images_with_false_positives": 1, "image_fp_rate": 1,
        })
        self.assertEqual(result["timing"]["processing_sec"], 6)
        self.assertEqual(result["timing"]["mean_processing_sec"], 2)
        self.assertEqual(result["timing"]["full_run"]["total_sec"], 100)
        self.assertEqual(result["groups"]["moths"]["n_images"], 2)
        self.assertEqual(len(result["predictions"]["sha256"]), 64)
        rows = [json.loads(line) for line in (self.output / "per_image.jsonl").read_text().splitlines()]
        self.assertEqual([row["processing_sec"] for row in rows], [1, 2, 3])
        with (self.output / "leaderboard.csv").open() as source:
            self.assertEqual(len(list(csv.DictReader(source))), 1)

    def test_timing_and_groups_are_optional(self):
        self.cfg.runs.test.run_path = None
        self.cfg.metadata_path = None
        self.assertEqual(evaluate.run(self.cfg), 0)
        result = self.report()["runs"]["test"]
        self.assertIsNone(result["timing"])
        self.assertEqual(result["groups"], {})

    def test_missing_reference_prediction_is_not_silently_excluded(self):
        self.write_jsonl(self.predictions, [row for row in self.prediction_rows if row["photo_id"] != 2])
        with self.assertRaisesRegex(ValueError, "missing 1 reference IDs"):
            evaluate.run(self.cfg)
        self.assertFalse(self.output.exists())

    def test_exact_scope_rejects_extra_predictions(self):
        self.cfg.scope = "exact"
        with self.assertRaisesRegex(ValueError, "scope=exact"):
            evaluate.run(self.cfg)

    def test_prediction_errors_and_invalid_boxes_do_not_produce_successful_only_metrics(self):
        for boxes, error in [(None, "failed inference"), ([{**BOX, "width": 0}], None)]:
            with self.subTest(error=error):
                self.cfg.output_dir = str(self.root / f"error-{error}")
                self.output = Path(self.cfg.output_dir)
                self.cfg.runs.test.run_path = None
                rows = [dict(row) for row in self.prediction_rows]
                rows[1].update(boxes=boxes, error=error)
                self.write_jsonl(self.predictions, rows)
                self.assertEqual(evaluate.run(self.cfg), 1)
                result = self.report()["runs"]["test"]
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["n_errors"], 1)
                self.assertIsNone(result["metrics"])
                self.assertIsNone(result["negative_images"])

    def test_error_in_reference_is_fatal(self):
        self.reference_rows[0]["error"] = "unreviewed"
        self.write_jsonl(self.reference, self.reference_rows)
        with self.assertRaisesRegex(ValueError, "Recorded processing error"):
            evaluate.run(self.cfg)

    def test_wrong_timing_folder_or_id_set_is_rejected(self):
        another = self.root / "another"
        another.mkdir()
        other_run = another / "run.json"
        other_run.write_text(json.dumps(self.timing))
        self.cfg.runs.test.run_path = str(other_run)
        with self.assertRaisesRegex(ValueError, "same folder"):
            evaluate.run(self.cfg)
        self.cfg.runs.test.run_path = str(self.run_path)
        self.timing["images"] = self.timing["images"][:-1]
        self.run_path.write_text(json.dumps(self.timing))
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            evaluate.run(self.cfg)

    def test_timing_box_counts_and_errors_must_match_predictions(self):
        for change in [{"scores": []}, {"error": "wrong run"}]:
            with self.subTest(change=change):
                info = json.loads(json.dumps(self.timing))
                info["images"][1].update(change)
                self.run_path.write_text(json.dumps(info))
                with self.assertRaisesRegex(ValueError, "match"):
                    evaluate.run(self.cfg)

    def test_missing_or_invalid_selected_time_is_not_fabricated(self):
        for seconds in [None, -1, float("nan"), True]:
            with self.subTest(seconds=seconds):
                info = json.loads(json.dumps(self.timing))
                info["images"][0]["processing_sec"] = seconds
                self.run_path.write_text(json.dumps(info))
                with self.assertRaisesRegex(ValueError, "processing_sec"):
                    evaluate.run(self.cfg)

    def test_correct_empty_answer_does_not_add_tp_or_f1_one(self):
        self.write_jsonl(self.reference, [{"photo_id": 3, "boxes": []}])
        self.write_jsonl(self.predictions, [{"photo_id": 3, "boxes": []}])
        self.cfg.runs.test.run_path = None
        self.assertEqual(evaluate.run(self.cfg), 0)
        result = self.report()["runs"]["test"]
        self.assertEqual(result["metrics"]["0.5"]["tp"], 0)
        self.assertIsNone(result["metrics"]["0.5"]["f1"])
        self.assertEqual(result["negative_images"]["image_fp_rate"], 0)

    def test_does_not_overwrite_reports(self):
        self.output.mkdir()
        (self.output / "metrics.json").write_text("existing")
        with self.assertRaises(FileExistsError):
            evaluate.run(self.cfg)
        self.assertEqual((self.output / "metrics.json").read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
