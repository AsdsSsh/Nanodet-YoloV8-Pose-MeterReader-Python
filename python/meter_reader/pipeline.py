import warnings
import json
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import MeterReaderConfig
from .geometry import compensated_value, scale_value
from .nanodet import NanoDet
from .ncnn_backend import NcnnBackend
from .pose import YoloV8Pose
from .scale_ocr import ScaleOcr
from .types import Detection, MeterReading


class MeterReader:
    def __init__(self, config: MeterReaderConfig) -> None:
        config.validate()
        self.config = config
        nanodet_backend = NcnnBackend(
            config.nanodet_param,
            config.nanodet_bin,
            config.use_gpu,
            config.num_threads,
        )
        pose_backend = NcnnBackend(
            config.pose_param,
            config.pose_bin,
            config.use_gpu,
            config.num_threads,
        )
        self.nanodet = NanoDet(nanodet_backend, config.nanodet)
        self.pose = YoloV8Pose(pose_backend, config.pose, config.pointer)
        self.ocr = ScaleOcr(config.ocr)

    def read(
        self,
        image: np.ndarray,
        apply_compensation: bool = True,
        debug: bool = False,
    ) -> List[MeterReading]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        readings = []
        for index, detection in enumerate(self.nanodet.detect(image)):
            if not self._valid_aspect_ratio(detection.width, detection.height):
                ratio = detection.width / detection.height
                warnings.warn(
                    "Skipping meter {} because ROI aspect ratio {:.3f} is outside the "
                    "configured range".format(index, ratio)
                )
                continue
            roi = self.nanodet.crop(image, detection)
            if roi.size == 0:
                warnings.warn("Skipping meter {} because its ROI is empty".format(index))
                continue
            try:
                reading = self._read_roi(roi, detection, apply_compensation, debug, index)
            except (RuntimeError, ValueError) as exc:
                # The detector box can cut off part of the dial, e.g. one scale
                # endpoint that the pose model needs. Retry on a padded crop
                # that shows the full dial before giving up.
                padded = self._padded_crop(image, detection)
                if padded is None:
                    warnings.warn("Failed to read meter {}: {}".format(index, exc))
                    continue
                padded_roi, padded_detection = padded
                try:
                    reading = self._read_roi(
                        padded_roi, padded_detection, apply_compensation, debug, index
                    )
                except (RuntimeError, ValueError):
                    warnings.warn("Failed to read meter {}: {}".format(index, exc))
                    continue
            readings.append(reading)
        return readings

    def _read_roi(
        self,
        roi: np.ndarray,
        detection: Detection,
        apply_compensation: bool,
        debug: bool,
        index: int,
    ) -> MeterReading:
        pose_detections = self.pose.detect(roi)
        if debug:
            print(
                "meter {} pose debug: {}".format(
                    index,
                    json.dumps(self.pose.last_debug, ensure_ascii=False),
                )
            )
        points = self.pose.points_from_detections(roi, pose_detections)
        effective_scale = self.config.scale
        scale_source = "config"
        if self.config.ocr.enabled:
            ocr_scale = self.ocr.read_scale(roi, pose_detections, self.config.scale)
            if ocr_scale is not None:
                effective_scale = ocr_scale
                scale_source = "ocr"
            if debug:
                print(
                    "meter {} ocr debug: {}".format(
                        index,
                        json.dumps(self.ocr.last_debug, ensure_ascii=False),
                    )
                )
        value = scale_value(points, effective_scale)
        display_value = compensated_value(
            value, effective_scale, apply_compensation
        )
        return MeterReading(
            detection,
            value,
            display_value,
            effective_scale.unit,
            points,
            effective_scale.beginning,
            effective_scale.end,
            scale_source,
        )

    def _padded_crop(
        self, image: np.ndarray, detection: Detection
    ) -> Optional[Tuple[np.ndarray, Detection]]:
        padding = self.config.roi_padding
        if padding <= 0.0:
            return None
        pad_x = int(detection.width * padding)
        pad_y = int(detection.height * padding)
        x1 = max(0, int(detection.x) - pad_x)
        y1 = max(0, int(detection.y) - pad_y)
        x2 = min(image.shape[1], int(detection.x + detection.width) + pad_x)
        y2 = min(image.shape[0], int(detection.y + detection.height) + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        if (x1, y1, x2, y2) == (
            int(detection.x),
            int(detection.y),
            int(detection.x + detection.width),
            int(detection.y + detection.height),
        ):
            return None
        expanded = Detection(
            x1, y1, x2 - x1, y2 - y1, detection.score, detection.label
        )
        return image[y1:y2, x1:x2].copy(), expanded

    def _valid_aspect_ratio(self, width: float, height: float) -> bool:
        if width <= 0.0 or height <= 0.0:
            return False
        threshold = self.config.aspect_ratio_threshold
        if threshold <= 0.0:
            return True
        ratio = width / height
        return threshold <= ratio <= 1.0 / threshold

    @staticmethod
    def visualize(image: np.ndarray, readings: List[MeterReading]) -> np.ndarray:
        output = image.copy()
        for reading in readings:
            detection = reading.detection
            origin_x = max(0, int(detection.x))
            origin_y = max(0, int(detection.y))
            x1, y1 = origin_x, origin_y
            x2 = min(output.shape[1] - 1, int(detection.x + detection.width))
            y2 = min(output.shape[0] - 1, int(detection.y + detection.height))
            cv2.rectangle(output, (x1, y1), (x2, y2), (237, 189, 101), 2)

            points = reading.points
            start = (int(points.start[0] + origin_x), int(points.start[1] + origin_y))
            end = (int(points.end[0] + origin_x), int(points.end[1] + origin_y))
            center = (int(points.center[0] + origin_x), int(points.center[1] + origin_y))
            pointer = (int(points.pointer[0] + origin_x), int(points.pointer[1] + origin_y))
            cv2.circle(output, start, 3, (15, 242, 235), -1)
            cv2.circle(output, end, 3, (15, 242, 235), -1)
            cv2.circle(output, center, 3, (0, 0, 255), -1)
            cv2.circle(output, pointer, 3, (255, 0, 0), -1)
            cv2.line(output, center, pointer, (255, 255, 0), 1, cv2.LINE_AA)

            # Label the scale range endpoints next to their keypoints.
            min_text = "min {:.3f}".format(reading.scale_begin)
            max_text = "max {:.3f}".format(reading.scale_end)
            min_origin = (
                max(0, min(start[0] - 10, output.shape[1] - 90)),
                min(output.shape[0] - 5, start[1] + 25),
            )
            max_origin = (
                max(0, min(end[0] - 80, output.shape[1] - 90)),
                min(output.shape[0] - 5, end[1] + 25),
            )
            cv2.putText(
                output,
                min_text,
                min_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (15, 242, 235),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                max_text,
                max_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (15, 242, 235),
                1,
                cv2.LINE_AA,
            )

            label = "Meter: {:.3f} {}".format(reading.display_value, reading.unit)
            text_y = max(20, y1 - 5)
            cv2.putText(
                output,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return output
