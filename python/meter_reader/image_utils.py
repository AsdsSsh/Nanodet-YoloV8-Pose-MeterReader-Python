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


def merge_duplicates(
    detections: List[Detection],
    threshold: float,
) -> List[Detection]:
    """Merge detections that overlap by at least `threshold` IoU into their
    bounding union, keeping the highest score. The detector can fire twice
    on one meter (e.g. two anchor scales each covering part of a dial that
    fills the frame); a single union box crops better for the reader than
    either partial box. Unlike NMS the merged box is the union, not the
    higher-scoring box, so the dial stays complete inside the crop."""
    merged: List[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        for group in merged:
            if box_iou(detection, [group])[0] >= threshold:
                x1 = min(detection.x, group.x)
                y1 = min(detection.y, group.y)
                x2 = max(detection.x + detection.width, group.x + group.width)
                y2 = max(detection.y + detection.height, group.y + group.height)
                group.x, group.y = x1, y1
                group.width = x2 - x1
                group.height = y2 - y1
                group.score = max(group.score, detection.score)
                break
        else:
            merged.append(detection)
    return merged



def find_dial_pointer(image):
    """Locate the needle of a round dial gauge by image processing alone
    (no model). This backs up the OCR scale-end fallback when the pose
    model cannot even find the pointer class, e.g. on full-circle dial
    indicators whose layout is outside the training domain.

    Steps: the dial is the largest Hough circle; the needle is the longest
    straight line through its center found on Canny edges of the dial face
    (raw gray -- global equalization destroys the needle contrast); the tip
    is the thinner end of the needle (dial-indicator needles carry a blunt
    counterweight opposite the pointed tip).

    Returns (center, tip, line) as ((x, y), (x, y), (x1, y1, x2, y2)) in
    image coordinates, or None when the needle cannot be found reliably.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        int(0.5 * min(height, width)),
        param1=100,
        param2=30,
        minRadius=int(0.2 * min(height, width)),
        maxRadius=int(0.6 * max(height, width)),
    )
    if circles is None:
        return None
    center_x, center_y, radius = max(circles[0], key=lambda c: c[2])
    crop_x1 = max(0, int(center_x - radius))
    crop_y1 = max(0, int(center_y - radius))
    crop_x2 = min(width, int(center_x + radius))
    crop_y2 = min(height, int(center_y + radius))
    if crop_x2 - crop_x1 < 40 or crop_y2 - crop_y1 < 40:
        return None
    dial = gray[crop_y1:crop_y2, crop_x1:crop_x2]
    up = cv2.resize(dial, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    edges = cv2.Canny(up, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        55,
        minLineLength=int(radius * 0.9),
        maxLineGap=8,
    )
    if lines is None:
        return None
    ccx = (center_x - crop_x1) * 2.0
    ccy = (center_y - crop_y1) * 2.0
    best = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < radius * 0.9:
            continue
        distance = abs(
            (x2 - x1) * (ccy - y1) - (y2 - y1) * (ccx - x1)
        ) / length
        if distance > 0.15 * radius:
            continue
        if best is None or length > best[0]:
            best = (length, x1, y1, x2, y2)
    if best is None:
        return None
    length, x1, y1, x2, y2 = best
    unit_x, unit_y = (x2 - x1) / length, (y2 - y1) / length
    dark = up < 110
    dark_height, dark_width = dark.shape

    def extent(direction_x, direction_y):
        # Walk outward from the center along the axis, tolerating a few
        # pixels of axis error (the needle shaft is only a couple of pixels
        # wide and can sit slightly off the fitted line) and small gaps
        # (the shaft can be fragmented at this resolution). Stops at the
        # first real break -- printed digits beyond the needle are separate
        # ink and must not extend the run.
        last, gap = 0, 0
        perp_x, perp_y = -direction_y, direction_x
        # The strip is checked on the 2x upscaled dial, so +/-13 there is
        # about +/-6.5 px at 1x -- wide enough for the thin shaft to sit a
        # few pixels off the fitted axis without escaping the strip.
        for d in range(1, int(1.2 * radius)):
            x0 = ccx + direction_x * d
            y0 = ccy + direction_y * d
            found = False
            for s in range(-13, 14):
                x, y = int(round(x0 + perp_x * s)), int(round(y0 + perp_y * s))
                if 0 <= x < dark_width and 0 <= y < dark_height and dark[y, x]:
                    found = True
                    break
            if found:
                last, gap = d, 0
            else:
                gap += 1
                if gap > 12:
                    break
        return last

    extent_a = extent(unit_x, unit_y)
    extent_b = extent(-unit_x, -unit_y)
    if extent_a < 0.18 * radius or extent_b < 0.18 * radius:
        return None
    # The needle's tip reaches the scale; the counterweight is the short
    # blunt end on the opposite side. The longer dark run is the tip.
    if abs(extent_a - extent_b) < 6:
        return None  # balanced needle: cannot tell tip from counterweight
    if extent_a >= extent_b:
        tip = (
            float(center_x + unit_x * extent_a),
            float(center_y + unit_y * extent_a),
        )
    else:
        tip = (
            float(center_x - unit_x * extent_b),
            float(center_y - unit_y * extent_b),
        )
    line = (
        float(x1) / 2.0 + crop_x1,
        float(y1) / 2.0 + crop_y1,
        float(x2) / 2.0 + crop_x1,
        float(y2) / 2.0 + crop_y1,
    )
    return (float(center_x), float(center_y)), tip, line
