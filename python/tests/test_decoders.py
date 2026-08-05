import importlib.util
import math
import sys
import types
import unittest

import numpy as np


if importlib.util.find_spec("cv2") is None:
    sys.modules["cv2"] = types.ModuleType("cv2")

from meter_reader.config import NanoDetConfig, PoseConfig, ScaleConfig
from meter_reader.geometry import angle_ratio, compensated_value
from meter_reader.nanodet import NanoDet
from meter_reader.pose import YoloV8Pose
from meter_reader.types import MeterPoints


class DecoderTests(unittest.TestCase):
    def test_nanodet_output_shape_and_dfl_decode(self):
        config = NanoDetConfig()
        row_count = sum(
            math.ceil(config.input_size[0] / stride)
            * math.ceil(config.input_size[1] / stride)
            for stride in config.strides
        )
        output = np.zeros((row_count, 33), dtype=np.float32)
        output[0, 0] = 0.9
        for side in range(4):
            start = config.num_classes + side * (config.reg_max + 1)
            output[0, start : start + config.reg_max + 1] = -10.0
            output[0, start + 2] = 10.0

        detector = NanoDet(None, config)
        detections = detector._decode(output)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].score, 0.9, places=5)
        self.assertGreater(detections[0].width, 0.0)

    def test_pose_output_shape_and_keypoint_decode(self):
        config = PoseConfig()
        output = np.full((2, 2, 73), -10.0, dtype=np.float32)
        row = output[0, 0]
        row[0] = 10.0
        for side in range(4):
            start = config.num_classes + side * config.reg_max
            row[start : start + config.reg_max] = -10.0
            row[start + 2] = 10.0
        keypoint_offset = config.num_classes + 4 * config.reg_max
        row[keypoint_offset : keypoint_offset + 6] = (0.25, 0.25, 10.0, 0.5, 0.5, 10.0)

        detector = YoloV8Pose(None, config, None)
        detections = detector._decode_output(output, stride=8)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, 0)
        self.assertEqual(len(detections[0].keypoints), 2)
        self.assertAlmostEqual(detections[0].keypoints[0][0], 4.0)

    def test_geometry_matches_cpp_formula(self):
        points = MeterPoints(
            start=(-1.0, 1.0),
            end=(1.0, 1.0),
            center=(0.0, 0.0),
            pointer=(0.0, -1.0),
        )
        self.assertAlmostEqual(angle_ratio(points), 0.5)
        self.assertAlmostEqual(compensated_value(0.5, ScaleConfig()), 0.512)

    def test_compensation_scales_with_range(self):
        # 0-25 range: midpoint 12.5, +1.2%*25 = +0.3 below it, +0.8%*25 = +0.2 above.
        scale = ScaleConfig(end=25.0)
        self.assertAlmostEqual(compensated_value(5.0, scale), 5.3)
        self.assertAlmostEqual(compensated_value(12.5, scale), 12.8)
        self.assertAlmostEqual(compensated_value(20.0, scale), 20.2)

    def test_compensation_disabled_returns_value(self):
        scale = ScaleConfig(end=25.0)
        self.assertAlmostEqual(compensated_value(5.0, scale, enabled=False), 5.0)

    def test_compensation_zero_range_returns_value(self):
        scale = ScaleConfig(beginning=1.0, end=1.0)
        self.assertAlmostEqual(compensated_value(1.0, scale), 1.0)


if __name__ == "__main__":
    unittest.main()
