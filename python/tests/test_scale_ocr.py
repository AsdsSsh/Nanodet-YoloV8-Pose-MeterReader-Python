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


class ReadScaleEndsTests(unittest.TestCase):
    """Tests for the whole-dial OCR fallback (read_scale_ends). The dial is
    synthetic: center (100, 100), numbers at radius 60, arc over the top from
    the min (left, math-frame angle 180 deg) to the max (right, 0 deg)."""

    CENTER = (100.0, 100.0)

    def _config(self, **overrides):
        return OcrConfig(upscale_factor=1.0, **overrides)

    def _box_at(self, x, y):
        return np.array(
            [[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]],
            dtype=np.float32,
        )

    def _token(self, text, angle_degrees, score=0.99, center=(100.0, 100.0)):
        x = center[0] + 60.0 * np.cos(np.radians(angle_degrees))
        y = center[1] - 60.0 * np.sin(np.radians(angle_degrees))
        return [self._box_at(x, y), text, score]

    def _read(self, results, config=None, center=None, pointer=(100.0, 40.0)):
        ocr = ScaleOcr(config or self._config())
        engine = FakeEngine([results])
        ocr.set_engine(engine)
        scale = ocr.read_scale_ends(
            _roi(), center or self.CENTER, pointer, ScaleConfig(unit="MPa")
        )
        return engine, scale

    def test_reads_range_and_interpolates_pointer(self):
        _, result = self._read(
            [
                self._token("0", 180),
                self._token("20", 144),
                self._token("40", 108),
                self._token("60", 72),
                self._token("80", 36),
                self._token("100", 0),
            ]
        )
        self.assertIsNotNone(result)
        start, end, scale, value = result
        self.assertAlmostEqual(start[0], 40.0, places=1)
        self.assertAlmostEqual(end[0], 160.0, places=1)
        self.assertAlmostEqual(scale.beginning, 0.0)
        self.assertAlmostEqual(scale.end, 100.0)
        self.assertAlmostEqual(value, 50.0, places=6)

    def test_piecewise_interpolation_handles_uneven_spacing(self):
        # The 60 is shifted toward the 40 (81 deg instead of 72), so the
        # straight arc ratio would read 50 but the local segment reads 53.3.
        _, result = self._read(
            [
                self._token("0", 180),
                self._token("20", 144),
                self._token("40", 108),
                self._token("60", 81),
                self._token("80", 36),
                self._token("100", 0),
            ]
        )
        self.assertIsNotNone(result)
        _start, _end, _scale, value = result
        self.assertAlmostEqual(value, 53.333, places=3)

    def test_rejects_low_confidence_tokens(self):
        engine, result = self._read(
            [
                self._token("0", 180),
                self._token("100", 0, score=0.5),
                self._token("99", 36, score=0.6),
            ]
        )
        # Only "0" survives the confidence floor; one token is not enough.
        self.assertIsNone(result)
        self.assertEqual(engine.calls, 1)

    def test_ignores_serial_number_tokens(self):
        _, result = self._read(
            [
                self._token("0", 180),
                self._token("100", 0),
                [self._box_at(100.0, 160.0), "80723048", 0.99],
            ]
        )
        self.assertIsNotNone(result)
        start, end, scale, value = result
        self.assertAlmostEqual(scale.end, 100.0)
        self.assertAlmostEqual(end[1], 100.0, places=1)

    def test_ccw_gauge_direction(self):
        # Mirrored dial: min at the right (0 deg), max at the left (180 deg).
        _, result = self._read(
            [
                self._token("0", 0),
                self._token("50", 90),
                self._token("100", 180),
            ],
            pointer=(100.0, 40.0),
        )
        self.assertIsNotNone(result)
        _start, _end, _scale, value = result
        self.assertAlmostEqual(value, 50.0, places=6)

    def test_full_circle_dial(self):
        # Dial indicator: numbers 10-90 around a full circle, the 0/100 gap
        # at the top of the dial. The top heuristic alone would pick the
        # short (wrong) arc; the intermediate numbers must vote for the long
        # arc, and the empty top gap must be inferred as the 0/100 position.
        center = (186.0, 188.0)
        _, result = self._read(
            [
                self._token("10", 55, center=center),
                self._token("20", 17, center=center),
                self._token("30", -18, center=center),
                self._token("40", -51, center=center),
                self._token("50", -87, center=center),
                self._token("60", -121, center=center),
                self._token("70", -154, center=center),
                self._token("80", 173, center=center),
                self._token("90", 130, center=center),
            ],
            pointer=(143.6, 230.4),  # math-frame angle -135: between 60 and 70
            center=center,
        )
        self.assertIsNotNone(result)
        start, end, scale, value = result
        # The 0/100 gap at the top is inferred, so the scale is 0-100.
        self.assertAlmostEqual(scale.beginning, 0.0)
        self.assertAlmostEqual(scale.end, 100.0)
        self.assertGreater(value, 60.0)
        self.assertLess(value, 70.0)
        # Both scale ends sit at the gap midpoint (the top of the dial).
        self.assertAlmostEqual(start[0], end[0], places=6)
        self.assertAlmostEqual(start[1], end[1], places=6)

    def test_center_label_tokens_filtered(self):
        # "0.01mm" and "0-50um" labels near the pivot parse as numbers; they
        # must not become scale ends.
        _, result = self._read(
            [
                self._token("0", 180),
                self._token("50", 90),
                self._token("100", 0),
                [self._box_at(100.0, 145.0), "0.01", 0.99],
                [self._box_at(100.0, 155.0), "0", 0.99],
            ],
            pointer=(100.0, 40.0),
        )
        self.assertIsNotNone(result)
        _start, _end, scale, _value = result
        self.assertAlmostEqual(scale.beginning, 0.0)
        self.assertAlmostEqual(scale.end, 100.0)

    def test_degenerate_range_returns_none(self):
        _, result = self._read(
            [self._token("100", 0), self._token("100", 180)]
        )
        self.assertIsNone(result)

    def test_disabled_returns_none(self):
        ocr = ScaleOcr(OcrConfig(enabled=False))
        engine = FakeEngine([])
        ocr.set_engine(engine)
        self.assertIsNone(
            ocr.read_scale_ends(
                _roi(), self.CENTER, (100.0, 40.0), ScaleConfig()
            )
        )
        self.assertEqual(engine.calls, 0)


if __name__ == "__main__":
    unittest.main()
