import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import PointerConfig, PoseConfig
from .image_utils import nms
from .ncnn_backend import NcnnBackend
from .types import MeterPoints, Point, PoseDetection


POSE_LABELS = ("pointer_rect", "left_rect", "right_rect")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


class YoloV8Pose:
    def __init__(self, backend: NcnnBackend, config: PoseConfig, pointer_config: PointerConfig) -> None:
        self.backend = backend
        self.config = config
        self.pointer_config = pointer_config
        self.last_debug = {}

    @staticmethod
    def _softmax_expectation(values: np.ndarray) -> float:
        values = values.astype(np.float32)
        values = values - np.max(values)
        probabilities = np.exp(values)
        probabilities /= max(float(np.sum(probabilities)), 1e-12)
        return float(np.dot(np.arange(len(values), dtype=np.float32), probabilities))

    def _as_grid(self, output: np.ndarray) -> np.ndarray:
        output = np.asarray(output, dtype=np.float32)
        output = np.squeeze(output)
        expected = self.config.num_classes + 4 * self.config.reg_max + 3 * self.config.keypoint_count
        if output.ndim != 3:
            raise ValueError("Unexpected YOLOv8-Pose output shape: {}".format(output.shape))
        if output.shape[-1] == expected:
            return output
        if output.shape[0] == expected:
            return output.transpose(1, 2, 0)
        raise ValueError("Unexpected YOLOv8-Pose output shape: {}".format(output.shape))

    def _decode_output(self, output: np.ndarray, stride: int) -> List[PoseDetection]:
        grid = self._as_grid(output)
        num_grid_y, num_grid_x, _ = grid.shape
        reg_max = self.config.reg_max
        class_count = self.config.num_classes
        detections: List[PoseDetection] = []
        for grid_y in range(num_grid_y):
            for grid_x in range(num_grid_x):
                row = grid[grid_y, grid_x]
                class_logits = np.clip(row[:class_count], -80.0, 80.0)
                class_scores = 1.0 / (1.0 + np.exp(-class_logits))
                label = int(np.argmax(class_scores))
                score = float(class_scores[label])
                if score < self.config.probability_threshold:
                    continue

                x0 = grid_x + 0.5 - self._softmax_expectation(row[class_count : class_count + reg_max])
                y0 = grid_y + 0.5 - self._softmax_expectation(row[class_count + reg_max : class_count + 2 * reg_max])
                x1 = grid_x + 0.5 + self._softmax_expectation(row[class_count + 2 * reg_max : class_count + 3 * reg_max])
                y1 = grid_y + 0.5 + self._softmax_expectation(row[class_count + 3 * reg_max : class_count + 4 * reg_max])
                x0, y0, x1, y1 = [value * stride for value in (x0, y0, x1, y1)]

                keypoints = []
                keypoint_offset = class_count + 4 * reg_max
                for keypoint in range(self.config.keypoint_count):
                    offset = keypoint_offset + keypoint * 3
                    keypoints.append(
                        (
                            (float(row[offset]) * 2.0 + grid_x) * stride,
                            (float(row[offset + 1]) * 2.0 + grid_y) * stride,
                            self._sigmoid(float(row[offset + 2])),
                        )
                    )
                detections.append(
                    PoseDetection(x0, y0, x1 - x0, y1 - y0, score, label, keypoints)
                )
        return detections

    def detect(self, image: np.ndarray) -> List[PoseDetection]:
        image_height, image_width = image.shape[:2]
        width, height = image_width, image_height
        if width > height:
            scale = float(self.config.target_size) / width
            width = self.config.target_size
            height = int(height * scale)
        else:
            scale = float(self.config.target_size) / height
            height = self.config.target_size
            width = int(width * scale)

        resized = cv2.resize(image, (width, height))
        width_pad = int(math.ceil(width / self.config.max_stride) * self.config.max_stride - width)
        height_pad = int(math.ceil(height / self.config.max_stride) * self.config.max_stride - height)
        top, bottom = height_pad // 2, height_pad - height_pad // 2
        left, right = width_pad // 2, width_pad - width_pad // 2
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_mat = self.backend.from_pixels(padded, "PIXEL_RGB")
        self.backend.normalize(input_mat, (0.0, 0.0, 0.0), (1.0 / 255.0,) * 3)
        outputs = self.backend.infer(self.config.input_name, input_mat, self.config.output_names)

        proposals: List[PoseDetection] = []
        output_shapes = []
        for output, stride in zip(outputs, self.config.output_strides):
            output_shapes.append(tuple(np.asarray(output).shape))
            proposals.extend(self._decode_output(output, stride))
        # Standard YOLOv8 NMS is per class. Class-agnostic NMS wrongly
        # suppresses a lower-scoring scale-end box when it overlaps the
        # (larger) pointer box or the other scale-end box, e.g. when the
        # pointer points at a scale end.
        selected = []
        for label in range(self.config.num_classes):
            selected.extend(
                nms(
                    [item for item in proposals if item.label == label],
                    self.config.nms_threshold,
                )
            )
        self.last_debug = {
            "input_shape": tuple(image.shape),
            "output_shapes": output_shapes,
            "proposal_counts": dict(Counter(item.label for item in proposals)),
            "selected_counts": dict(Counter(item.label for item in selected)),
            "selected": [
                {
                    "label": POSE_LABELS[item.label],
                    "score": round(item.score, 4),
                    "box": [round(item.x, 1), round(item.y, 1), round(item.width, 1), round(item.height, 1)],
                }
                for item in selected
            ],
        }

        converted: List[PoseDetection] = []
        for detection in selected:
            x1, y1, x2, y2 = detection.xyxy
            x1 = _clamp((x1 - left) / scale, 0.0, float(image_width))
            y1 = _clamp((y1 - top) / scale, 0.0, float(image_height))
            x2 = _clamp((x2 - left) / scale, 0.0, float(image_width))
            y2 = _clamp((y2 - top) / scale, 0.0, float(image_height))
            keypoints = [
                (
                    _clamp((x - left) / scale, 0.0, float(image_width)),
                    _clamp((y - top) / scale, 0.0, float(image_height)),
                    score,
                )
                for x, y, score in detection.keypoints
            ]
            converted.append(
                PoseDetection(x1, y1, x2 - x1, y2 - y1, detection.score, detection.label, keypoints)
            )
        return converted

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def pointer_line(
        self, image: np.ndarray, detection: PoseDetection
    ) -> Optional[Tuple[int, int, int, int]]:
        x1 = max(0, int(detection.x))
        y1 = max(0, int(detection.y))
        x2 = min(image.shape[1], int(detection.x + detection.width))
        y2 = min(image.shape[0], int(detection.y + detection.height))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = image[y1:y2, x1:x2]
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        inverted = cv2.bitwise_not(equalized)
        blurred = cv2.medianBlur(inverted, 3)
        eroded = cv2.erode(blurred, kernel)
        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
        _, thresholded = cv2.threshold(
            closed, self.pointer_config.binary_threshold, 255, cv2.THRESH_BINARY
        )
        thinned = self._thin(thresholded)
        lines = cv2.HoughLinesP(
            thinned,
            1,
            np.pi / 180.0,
            self.pointer_config.hough_threshold,
            minLineLength=self.pointer_config.min_line_length,
            maxLineGap=self.pointer_config.max_line_gap,
        )
        if lines is None or len(lines) == 0:
            return None
        line = lines[0][0]
        return (
            int(line[0] + x1),
            int(line[1] + y1),
            int(line[2] + x1),
            int(line[3] + y1),
        )

    @staticmethod
    def _thin(source: np.ndarray) -> np.ndarray:
        try:
            thinning = cv2.ximgproc.thinning
        except AttributeError:
            thinning = None
        if thinning is not None:
            return thinning(source)

        image = (source > 0).astype(np.uint8)
        changed = True
        while changed:
            changed = False
            for phase in (0, 1):
                remove = []
                for y in range(1, image.shape[0] - 1):
                    for x in range(1, image.shape[1] - 1):
                        if image[y, x] == 0:
                            continue
                        neighbors = [
                            image[y - 1, x],
                            image[y - 1, x + 1],
                            image[y, x + 1],
                            image[y + 1, x + 1],
                            image[y + 1, x],
                            image[y + 1, x - 1],
                            image[y, x - 1],
                            image[y - 1, x - 1],
                        ]
                        transitions = sum(
                            neighbors[index] == 0
                            and neighbors[(index + 1) % 8] == 1
                            for index in range(8)
                        )
                        count = sum(neighbors)
                        if not (2 <= count <= 6 and transitions == 1):
                            continue
                        if phase == 0:
                            condition = (
                                neighbors[0] * neighbors[2] * neighbors[4] == 0
                                and neighbors[2] * neighbors[4] * neighbors[6] == 0
                            )
                        else:
                            condition = (
                                neighbors[0] * neighbors[2] * neighbors[6] == 0
                                and neighbors[0] * neighbors[4] * neighbors[6] == 0
                            )
                        if condition:
                            remove.append((y, x))
                if remove:
                    changed = True
                    for y, x in remove:
                        image[y, x] = 0
        return image * 255

    def pointer_points(
        self, image: np.ndarray, detections: Iterable[PoseDetection]
    ) -> Optional[Tuple[Point, Point, Optional[Tuple[int, int, int, int]]]]:
        """Extract (center, pointer, pointer_line) from the best pointer
        detection, refining the tip with the Hough line when available.
        Returns None when no usable pointer detection exists."""
        best = None
        for detection in detections:
            if detection.label == 0 and (
                best is None or detection.score > best.score
            ):
                best = detection
        if best is None:
            return None
        center = self._keypoint(best, 0)
        pointer = self._keypoint(best, 1)
        line = self.pointer_line(image, best)
        if line is not None:
            pointer = self._far_endpoint(center, line)
        return center, pointer, line

    def points_from_detections(
        self, image: np.ndarray, detections: Iterable[PoseDetection]
    ) -> MeterPoints:
        by_label: Dict[int, PoseDetection] = {}
        for detection in detections:
            if detection.label not in by_label or detection.score > by_label[detection.label].score:
                by_label[detection.label] = detection
        if not all(label in by_label for label in range(3)):
            raise ValueError(
                "Pose model did not find pointer, left endpoint, and right endpoint"
            )

        center, pointer, line = self.pointer_points(image, detections)
        start = self._last_valid_keypoint(by_label[1])
        end = self._last_valid_keypoint(by_label[2])
        start_angle = math.atan2(center[1] - start[1], start[0] - center[0])
        end_angle = math.atan2(center[1] - end[1], end[0] - center[0])
        separation = abs(
            (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
        )
        if separation < math.radians(self.config.min_arc_degrees):
            # Two scale ends at (nearly) the same spot around the dial is
            # impossible on a real gauge; the model is confused (it happens
            # on unusual layouts where it re-labels one end twice). Treat it
            # as a detection failure so the caller can fall back.
            raise ValueError(
                "Scale end points are too close together ({:.1f} degrees)".format(
                    math.degrees(separation)
                )
            )
        return MeterPoints(start, end, center, pointer, line)

    def _keypoint(self, detection: PoseDetection, index: int) -> Point:
        if index >= len(detection.keypoints):
            raise ValueError("Pose output is missing keypoint {}".format(index))
        point = detection.keypoints[index]
        if point[2] <= self.config.keypoint_threshold:
            raise ValueError("A required pose keypoint is below confidence threshold")
        return float(point[0]), float(point[1])

    def _last_valid_keypoint(self, detection: PoseDetection) -> Point:
        valid = [
            point
            for point in detection.keypoints
            if point[2] > self.config.keypoint_threshold
        ]
        if not valid:
            raise ValueError("A required pose keypoint is below confidence threshold")
        point = valid[-1]
        return float(point[0]), float(point[1])

    @staticmethod
    def _far_endpoint(center: Point, line: Tuple[int, int, int, int]) -> Point:
        first = (float(line[0]), float(line[1]))
        second = (float(line[2]), float(line[3]))
        first_distance = math.hypot(first[0] - center[0], first[1] - center[1])
        second_distance = math.hypot(second[0] - center[0], second[1] - center[1])
        return first if first_distance > second_distance else second
