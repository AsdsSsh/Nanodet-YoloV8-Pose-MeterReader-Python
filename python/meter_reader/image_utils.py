from typing import List, Sequence

import cv2
import numpy as np

from .types import Detection


def letterbox_black(image: np.ndarray, size: int):
    height, width = image.shape[:2]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if width == height:
        resized = cv2.resize(image, (size, size))
        canvas[:] = resized
        return canvas, (0, 0, size, size)

    if width > height:
        resized_width = size
        resized_height = int(np.floor(size * height / width))
        offset_x = 0
        offset_y = int(np.floor((size - resized_height) / 2.0))
    else:
        resized_height = size
        resized_width = int(np.floor(size * width / height))
        offset_x = int(np.floor((size - resized_width) / 2.0))
        offset_y = 0

    resized = cv2.resize(image, (resized_width, resized_height))
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas, (offset_x, offset_y, resized_width, resized_height)


def box_iou(
    reference: Detection,
    candidates: Sequence[Detection],
    inclusive: bool = False,
) -> np.ndarray:
    if not candidates:
        return np.empty((0,), dtype=np.float32)
    ref_x1, ref_y1, ref_x2, ref_y2 = reference.xyxy
    result = []
    edge = 1.0 if inclusive else 0.0
    ref_area = max(0.0, ref_x2 - ref_x1 + edge) * max(
        0.0, ref_y2 - ref_y1 + edge
    )
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.xyxy
        area = max(0.0, x2 - x1 + edge) * max(0.0, y2 - y1 + edge)
        inter_w = max(0.0, min(ref_x2, x2) - max(ref_x1, x1) + edge)
        inter_h = max(0.0, min(ref_y2, y2) - max(ref_y1, y1) + edge)
        intersection = inter_w * inter_h
        result.append(intersection / max(ref_area + area - intersection, 1e-12))
    return np.asarray(result, dtype=np.float32)


def nms(
    detections: List[Detection],
    threshold: float,
    inclusive: bool = False,
) -> List[Detection]:
    remaining = sorted(detections, key=lambda item: item.score, reverse=True)
    selected = []
    while remaining:
        best = remaining.pop(0)
        selected.append(best)
        overlaps = box_iou(best, remaining, inclusive=inclusive)
        remaining = [
            item for item, overlap in zip(remaining, overlaps) if overlap < threshold
        ]
    return selected
