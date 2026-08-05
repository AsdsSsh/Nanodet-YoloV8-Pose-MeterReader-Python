import importlib.util
import sys
import types
import unittest
import warnings

import numpy as np


if importlib.util.find_spec("cv2") is None:
    sys.modules["cv2"] = types.ModuleType("cv2")

from meter_reader.config import OcrConfig, ScaleConfig
from meter_reader.scale_ocr import (
    ScaleOcr,
    expand_rect,
    parse_number,
    select_number,
)
from meter_reader.types import PoseDetection


def _pose(label, score=0.9, x=10.0, y=10.0, width=20.0, height=20.0):
    return PoseDetection(x, y, width, height, score, label, [])


def _roi(width=100, height=100):
    return np.zeros((height, width, 3), dtype=np.uint8)


class V2Result:
    """Mimics the result object of the newer `rapidocr` package."""

    def __init__(self, txts, scores):
        self.txts = txts
        self.scores = scores


class FakeEngine:
    """Consumes results from a sequence, one per engine call, in call order.
    The real call order is left-crop native, left-crop upscaled, right-crop
    native, right-crop upscaled (upscale skipped when configured off)."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, crop):
        result = self.results[self.calls % len(self.results)]
        self.calls += 1
        return (result, 0.0)


class ParseNumberTests(unittest.TestCase):
    def test_extracts_digits(self):
        self.assertAlmostEqual(parse_number("1.6"), 1.6)
        self.assertAlmostEqual(parse_number("-0.1"), -0.1)
        self.assertAlmostEqual(parse_number("0"), 0.0)
        self.assertAlmostEqual(parse_number("0."), 0.0)

    def test_ignores_units_and_empty_text(self):
        self.assertIsNone(parse_number("MPa"))
        self.assertIsNone(parse_number("kg"))
        self.assertIsNone(parse_number(""))
        self.assertAlmostEqual(parse_number("0.1MPa"), 0.1)

    def test_unicode_minus(self):
        self.assertAlmostEqual(parse_number("−0.5"), -0.5)

    def test_comma_decimal(self):
        self.assertAlmostEqual(parse_number("1,6"), 1.6)


class SelectNumberTests(unittest.TestCase):
    def test_prefers_highest_score(self):
        self.assertAlmostEqual(
            select_number([("MPa", 0.99), ("1.6", 0.95)]), 1.6
        )
        self.assertAlmostEqual(
            select_number([("abc", 0.9), ("-0.1", 0.8)]), -0.1
        )

    def test_no_score_keeps_order(self):
        self.assertAlmostEqual(
            select_number([(None, None), ("1.6", None)]), 1.6
        )
        self.assertAlmostEqual(
            select_number([("0", None), ("1.6", None)]), 0.0
        )

    def test_none_when_no_number(self):
        self.assertIsNone(select_number([("MPa", 0.9), ("kg", 0.8)]))


class ExpandRectTests(unittest.TestCase):
    def test_clips_to_image(self):
        self.assertEqual(
            expand_rect(0.0, 0.0, 10.0, 10.0, 0.5, 100, 100, 1),
            (0, 0, 15, 15),
        )
        self.assertEqual(
            expand_rect(95.0, 95.0, 10.0, 10.0, 0.5, 100, 100, 1),
            (90, 90, 100, 100),
        )

    def test_degenerate_returns_none(self):
        self.assertIsNone(expand_rect(0.0, 0.0, 0.0, 10.0, 0.5, 100, 100, 1))
        self.assertIsNone(expand_rect(0.0, 0.0, 2.0, 2.0, 0.0, 100, 100, 16))

    def test_zero_padding_unchanged(self):
        self.assertEqual(
            expand_rect(10.0, 10.0, 20.0, 20.0, 0.0, 100, 100, 1),
            (10, 10, 30, 30),
        )


class ReadScaleTests(unittest.TestCase):
    def setUp(self):
        self.roi = _roi()
        self.detections = [_pose(1), _pose(2)]
        self.fallback = ScaleConfig(unit="MPa")
        self.config = OcrConfig(upscale_factor=1.0)

    def read_with(self, left_result, right_result):
        engine = FakeEngine([left_result, right_result])
        ocr = ScaleOcr(self.config)
        ocr.set_engine(engine)
        result = ocr.read_scale(self.roi, self.detections, self.fallback)
        return engine, result

    def test_v1_result_shape(self):
        box = np.zeros((4, 2), dtype=np.float32)
        engine, scale = self.read_with(
            [[box, "0.0", 0.9], [box, "MPa", 0.99]], [[box, "1.6", 0.9]]
        )
        self.assertEqual(engine.calls, 2)
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale.beginning, 0.0)
        self.assertAlmostEqual(scale.end, 1.6)
        self.assertEqual(scale.unit, "MPa")
        self.assertEqual(scale.compensation_split, self.fallback.compensation_split)
        self.assertEqual(scale.lower_compensation, self.fallback.lower_compensation)
        self.assertEqual(scale.upper_compensation, self.fallback.upper_compensation)

    def test_v2_result_object_shape(self):
        engine, scale = self.read_with(
            V2Result(["0", "MPa"], [0.9, 0.99]), V2Result(["1.6"], [0.9])
        )
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale.beginning, 0.0)
        self.assertAlmostEqual(scale.end, 1.6)

    def test_no_text_returns_none(self):
        engine, scale = self.read_with(None, None)
        self.assertEqual(engine.calls, 2)
        self.assertIsNone(scale)

    def test_dual_scale_merges_candidates(self):
        # The upscaled read misreads "16" as "91" with a lower score; the
        # native read is correct and more confident, so it must win.
        config = OcrConfig(upscale_factor=2.0)
        box = np.zeros((4, 2), dtype=np.float32)
        results = [
            [[box, "0", 0.95]],  # left native
            [[box, "0", 0.9]],  # left upscaled
            [[box, "91", 0.87]],  # right native
            [[box, "16", 0.99]],  # right upscaled
        ]
        engine = FakeEngine(results)
        ocr = ScaleOcr(config)
        ocr.set_engine(engine)
        scale = ocr.read_scale(self.roi, self.detections, self.fallback)
        self.assertEqual(engine.calls, 4)
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale.end, 16.0)

    def test_reversed_range_returns_none(self):
        _, scale = self.read_with([[None, "1.6", 0.9]], [[None, "0.0", 0.9]])
        self.assertIsNone(scale)

    def test_equal_values_returns_none(self):
        _, scale = self.read_with([[None, "0.0", 0.9]], [[None, "0.0", 0.9]])
        self.assertIsNone(scale)

    def test_cache_hit_skips_engine(self):
        box = np.zeros((4, 2), dtype=np.float32)
        ocr = ScaleOcr(self.config)
        engine = FakeEngine([[[box, "0.0", 0.9]], [[box, "1.6", 0.9]]])
        ocr.set_engine(engine)
        first = ocr.read_scale(self.roi, self.detections, self.fallback)
        second = ocr.read_scale(self.roi, self.detections, self.fallback)
        self.assertEqual(engine.calls, 2)
        self.assertAlmostEqual(first.beginning, second.beginning)
        self.assertAlmostEqual(first.end, second.end)
        self.assertTrue(ocr.last_debug["cache_hit"])

    def test_cache_eviction(self):
        config = OcrConfig(upscale_factor=1.0, cache_max_entries=2)
        ocr = ScaleOcr(config)
        engine = FakeEngine(
            [[[None, "0.0", 0.9]], [[None, "1.6", 0.9]]]
        )
        ocr.set_engine(engine)
        keys = [
            [_pose(1, x=10.0), _pose(2, x=50.0)],
            [_pose(1, x=20.0), _pose(2, x=60.0)],
            [_pose(1, x=30.0), _pose(2, x=70.0)],
        ]
        for detections in keys:
            ocr.read_scale(self.roi, detections, self.fallback)
        # The first key was evicted, so this read hits the engine again.
        ocr.read_scale(self.roi, keys[0], self.fallback)
        self.assertEqual(engine.calls, 8)

    def test_disabled_returns_none(self):
        ocr = ScaleOcr(OcrConfig(enabled=False))
        engine = FakeEngine([])
        ocr.set_engine(engine)
        self.assertIsNone(ocr.read_scale(self.roi, self.detections, self.fallback))
        self.assertEqual(engine.calls, 0)

    def test_lazy_engine_not_loaded(self):
        ocr = ScaleOcr(self.config)
        engine = FakeEngine([])
        ocr.set_engine(engine)
        self.assertEqual(engine.calls, 0)

    def test_missing_pose_label_returns_none(self):
        ocr = ScaleOcr(self.config)
        engine = FakeEngine([])
        ocr.set_engine(engine)
        self.assertIsNone(
            ocr.read_scale(self.roi, [_pose(1)], self.fallback)
        )
        self.assertEqual(engine.calls, 0)

    def test_engine_import_failure_warns_once(self):
        ocr = ScaleOcr(self.config)

        def fail():
            raise ImportError("no rapidocr installed")

        ocr._ensure_engine = fail
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(
                ocr.read_scale(self.roi, self.detections, self.fallback)
            )
            self.assertIsNone(
                ocr.read_scale(self.roi, self.detections, self.fallback)
            )
        self.assertEqual(len(caught), 1)
        self.assertIn("OCR", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
