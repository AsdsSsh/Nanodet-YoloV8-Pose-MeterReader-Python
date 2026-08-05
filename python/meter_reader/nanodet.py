from typing import List, Tuple

import cv2
import numpy as np

from .config import NanoDetConfig
from .image_utils import letterbox_black, merge_duplicates, nms
from .ncnn_backend import NcnnBackend
from .types import Detection


class NanoDet:
    """NanoDet detector matching the NCNN model used by the C++ version."""

    def __init__(self, backend: NcnnBackend, config: NanoDetConfig) -> None:
        self.backend = backend
        self.config = config

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        values = values - np.max(values)
        exp_values = np.exp(values)
        return exp_values / max(float(np.sum(exp_values)), 1e-12)

    def _decode(self, output: np.ndarray) -> List[Detection]:
        output = np.asarray(output, dtype=np.float32)
        output = np.squeeze(output)
        if output.ndim != 2:
            raise ValueError("Unexpected NanoDet output shape: {}".format(output.shape))
        if output.shape[1] != self.config.num_classes + 4 * (self.config.reg_max + 1):
            if output.shape[0] == self.config.num_classes + 4 * (self.config.reg_max + 1):
                output = output.T
            else:
                raise ValueError("Unexpected NanoDet output shape: {}".format(output.shape))

        priors = []
        input_height, input_width = self.config.input_size
        for stride in self.config.strides:
            feat_w = int(np.ceil(float(input_width) / stride))
            feat_h = int(np.ceil(float(input_height) / stride))
            for y in range(feat_h):
                for x in range(feat_w):
                    priors.append((x, y, stride))
        if len(priors) != output.shape[0]:
            raise ValueError(
                "NanoDet prior count {} does not match output rows {}".format(
                    len(priors), output.shape[0]
                )
            )

        results: List[Detection] = []
        bins = self.config.reg_max + 1
        for row, (grid_x, grid_y, stride) in zip(output, priors):
            scores = row[: self.config.num_classes]
            label = int(np.argmax(scores))
            score = float(scores[label])
            if score <= self.config.score_threshold:
                continue

            distances = []
            for side in range(4):
                distribution = self._softmax(
                    row[self.config.num_classes + side * bins :
                        self.config.num_classes + (side + 1) * bins]
                )
                distances.append(float(np.dot(np.arange(bins), distribution) * stride))

            center_x = grid_x * stride
            center_y = grid_y * stride
            x1 = max(center_x - distances[0], 0.0)
            y1 = max(center_y - distances[1], 0.0)
            x2 = min(center_x + distances[2], float(input_width))
            y2 = min(center_y + distances[3], float(input_height))
            results.append(Detection(x1, y1, x2 - x1, y2 - y1, score, label))
        return merge_duplicates(
            nms(results, self.config.nms_threshold, inclusive=True),
            self.config.merge_threshold,
        )

    def detect(self, image: np.ndarray) -> List[Detection]:
        resized, effect = letterbox_black(image, self.config.input_size[0])
        input_mat = self.backend.from_pixels(resized, "PIXEL_BGR")
        self.backend.normalize(input_mat, self.config.mean, self.config.norm)
        output = self.backend.infer(
            self.config.input_name, input_mat, (self.config.output_name,)
        )[0]
        boxes = self._decode(output)
        return self._convert_boxes(boxes, effect, image.shape[1], image.shape[0])

    @staticmethod
    def _convert_boxes(
        boxes: List[Detection],
        effect: Tuple[int, int, int, int],
        source_width: int,
        source_height: int,
    ) -> List[Detection]:
        offset_x, offset_y, effective_width, effective_height = effect
        width_ratio = float(source_width) / max(effective_width, 1)
        height_ratio = float(source_height) / max(effective_height, 1)
        converted = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            x1 = (x1 - offset_x) * width_ratio
            y1 = (y1 - offset_y) * height_ratio
            x2 = (x2 - offset_x) * width_ratio
            y2 = (y2 - offset_y) * height_ratio
            converted.append(
                Detection(x1, y1, x2 - x1, y2 - y1, box.score, box.label)
            )
        return converted

    @staticmethod
    def crop(image: np.ndarray, detection: Detection) -> np.ndarray:
        x1 = max(0, int(detection.x))
        y1 = max(0, int(detection.y))
        x2 = min(image.shape[1], int(detection.x + detection.width))
        y2 = min(image.shape[0], int(detection.y + detection.height))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=image.dtype)
        return image[y1:y2, x1:x2].copy()
