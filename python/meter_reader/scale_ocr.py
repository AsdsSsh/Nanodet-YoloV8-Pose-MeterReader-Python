import math
import re
import warnings
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import OcrConfig, ScaleConfig
from .types import Point, PoseDetection

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)")

CacheKey = Tuple[
    Tuple[int, int],
    Tuple[int, int, int, int],
    Tuple[int, int, int, int],
]


def parse_number(text: str) -> Optional[float]:
    """Extract the first number from OCR text. Unicode minus and comma
    decimal separators are normalized. Returns None when there is no number
    (e.g. "MPa", "kg" or empty text)."""
    normalized = text.replace("−", "-").replace(",", ".")
    match = _NUMBER_PATTERN.search(normalized)
    if match is None:
        return None
    return float(match.group())


def select_number(candidates: Sequence[Tuple[str, Optional[float]]]) -> Optional[float]:
    """Pick the most confident numeric token from one OCR crop. A crop can
    contain several text lines ("0", "MPa", stray ticks); the number with the
    highest score wins, and tokens without a score keep their original order
    (treated as -1)."""
    ranked = sorted(
        candidates,
        key=lambda item: -1.0 if item[1] is None else item[1],
        reverse=True,
    )
    for text, _score in ranked:
        if not isinstance(text, str):
            continue
        number = parse_number(text)
        if number is not None:
            return number
    return None


def expand_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    padding: float,
    image_width: int,
    image_height: int,
    min_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Grow a rect by `padding` times its own size on every side and clip it
    to the image. Returns None for degenerate boxes or crops smaller than
    min_size."""
    if width <= 0.0 or height <= 0.0:
        return None
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(image_width, int(x + width) + pad_x)
    y2 = min(image_height, int(y + height) + pad_y)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return x1, y1, x2, y2


class ScaleOcr:
    """Reads the meter's scale range by OCR-ing the two scale-end boxes that
    the pose model already detected (labels 1=left_rect, 2=right_rect). The
    engine is loaded lazily on the first use so that constructing this class
    (or importing the module) is free and safe without rapidocr installed."""

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self._engine = None
        self._engine_failed = False
        self._cache: "OrderedDict[CacheKey, Tuple[float, float]]" = OrderedDict()
        self.last_debug: dict = {}

    def set_engine(self, engine) -> None:
        """Inject a fake engine for tests; replaces the lazy loading path."""
        self._engine = engine

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        # Import inside the function: the module must be importable even when
        # rapidocr is not installed.
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        return self._engine

    @staticmethod
    def _normalize_output(output) -> List[Tuple[str, Optional[float], float, float]]:
        """Normalize the engine return value to a list of
        (text, score, center_x, center_y) entries in image coordinates.
        Handles both the v1 style (([box, txt, score], elapse) or None) and
        the newer rapidocr package style (an object with .txts/.scores)."""
        if isinstance(output, tuple) and len(output) == 2:
            result = output[0]
        else:
            result = output
        if result is None:
            return []
        def _box_center(box):
            try:
                points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                return float(points[:, 0].mean()), float(points[:, 1].mean())
            except (TypeError, ValueError):
                return 0.0, 0.0

        if hasattr(result, "txts"):
            texts = list(result.txts)
            scores = list(result.scores)
            boxes = getattr(result, "boxes", None)
            items = []
            for index, text in enumerate(texts):
                score = scores[index] if index < len(scores) else None
                box = boxes[index] if boxes is not None and index < len(boxes) else None
                center_x, center_y = _box_center(box)
                items.append(
                    (
                        str(text),
                        float(score) if score is not None else None,
                        center_x,
                        center_y,
                    )
                )
            return items
        items = []
        for entry in result:
            if len(entry) < 3:
                continue
            text, score = entry[1], entry[2]
            if not isinstance(text, str):
                continue
            center_x, center_y = _box_center(entry[0])
            items.append(
                (
                    text,
                    float(score) if score is not None else None,
                    center_x,
                    center_y,
                )
            )
        return items

    def _ocr_items(self, crop: np.ndarray) -> List[Tuple[str, Optional[float]]]:
        """OCR the crop at its native resolution and, when configured, at an
        upscaled resolution too. Results are merged and the caller picks the
        most confident number, so a misread caused by upscaling cannot win
        over a correct native read."""
        return [(text, score) for text, score, _x, _y in self._ocr_items_with_boxes(crop)]

    def _ocr_items_with_boxes(
        self, crop: np.ndarray
    ) -> List[Tuple[str, Optional[float], float, float]]:
        """Like _ocr_items but also reports each text's center position in
        the (un-upscaled) crop."""
        engine = self._ensure_engine()
        items = self._normalize_output(engine(crop))
        upscale = self.config.upscale_factor
        if upscale > 1.0 and max(crop.shape[0], crop.shape[1]) < 640:
            upscaled = cv2.resize(
                crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
            )
            items.extend(
                (text, score, x / upscale, y / upscale)
                for text, score, x, y in self._normalize_output(engine(upscaled))
            )
        return items

    @staticmethod
    def _quantize(value: float, step: int) -> int:
        if step <= 0:
            return int(value)
        return int(value / step)

    def _cache_key(
        self, roi: np.ndarray, left: PoseDetection, right: PoseDetection
    ) -> CacheKey:
        step = self.config.cache_quantize
        left_key = (
            self._quantize(left.x, step),
            self._quantize(left.y, step),
            self._quantize(left.width, step),
            self._quantize(left.height, step),
        )
        right_key = (
            self._quantize(right.x, step),
            self._quantize(right.y, step),
            self._quantize(right.width, step),
            self._quantize(right.height, step),
        )
        return (roi.shape[0], roi.shape[1]), left_key, right_key

    @staticmethod
    def _best_by_label(detections: Iterable[PoseDetection], label: int) -> Optional[PoseDetection]:
        best = None
        for detection in detections:
            if detection.label == label and (
                best is None or detection.score > best.score
            ):
                best = detection
        return best

    def _crop(self, roi: np.ndarray, box: PoseDetection) -> Optional[np.ndarray]:
        rect = expand_rect(
            box.x,
            box.y,
            box.width,
            box.height,
            self.config.crop_padding,
            roi.shape[1],
            roi.shape[0],
            self.config.min_crop_size,
        )
        if rect is None:
            return None
        x1, y1, x2, y2 = rect
        return roi[y1:y2, x1:x2]

    @staticmethod
    def _build_scale(beginning: float, end: float, fallback: ScaleConfig) -> ScaleConfig:
        return ScaleConfig(
            beginning=beginning,
            end=end,
            unit=fallback.unit,
            compensation_split=fallback.compensation_split,
            lower_compensation=fallback.lower_compensation,
            upper_compensation=fallback.upper_compensation,
            tilt_compensation=fallback.tilt_compensation,
        )

    def read_scale(
        self,
        roi: np.ndarray,
        pose_detections: Iterable[PoseDetection],
        fallback: ScaleConfig,
    ) -> Optional[ScaleConfig]:
        """OCR the two scale-end boxes and return the detected scale range, or
        None when it cannot be read reliably (caller falls back to the static
        config scale). Never raises: engine failures warn once and return
        None."""
        self.last_debug = {"cache_hit": False}
        if not self.config.enabled:
            return None
        detections = list(pose_detections)
        left = self._best_by_label(detections, 1)
        right = self._best_by_label(detections, 2)
        if left is None or right is None:
            return None

        key = self._cache_key(roi, left, right)
        if self.config.use_cache and key in self._cache:
            beginning, end = self._cache[key]
            self.last_debug["cache_hit"] = True
            return self._build_scale(beginning, end, fallback)

        try:
            left_crop = self._crop(roi, left)
            right_crop = self._crop(roi, right)
            if left_crop is None or right_crop is None:
                return None
            left_items = self._ocr_items(left_crop)
            right_items = self._ocr_items(right_crop)
        except (ImportError, RuntimeError, ValueError) as exc:
            if not self._engine_failed:
                self._engine_failed = True
                warnings.warn("OCR scale detection unavailable: {}".format(exc))
            return None

        self.last_debug["left_texts"] = [item[0] for item in left_items]
        self.last_debug["right_texts"] = [item[0] for item in right_items]
        beginning = select_number(left_items)
        end = select_number(right_items)
        self.last_debug["parsed"] = {"beginning": beginning, "end": end}

        if beginning is None or end is None:
            return None
        if not (math.isfinite(beginning) and math.isfinite(end)):
            return None
        if beginning >= end:
            # Never guess: an inverted or degenerate range is a detection
            # error, and the static config scale is the safer answer.
            return None

        if self.config.use_cache:
            self._cache[key] = (beginning, end)
            while len(self._cache) > self.config.cache_max_entries:
                self._cache.popitem(last=False)
        return self._build_scale(beginning, end, fallback)

    # Confidence floor for whole-dial OCR tokens used as scale ends. Low
    # scores are usually misreads (e.g. "2司" for "2"), and a wrong end
    # position is worse than falling back to the static config range.
    _MIN_END_SCORE = 0.8
    # A real dial arc spans roughly 90-270 degrees; anything outside is a
    # degenerate token pair (e.g. a bezel number next to a scale number).
    _MIN_ARC = math.pi / 3.0
    _MAX_ARC = 5.0 * math.pi / 3.0

    def read_scale_ends(
        self,
        roi: np.ndarray,
        center: Point,
        pointer: Point,
        fallback: ScaleConfig,
    ) -> Optional[Tuple[Point, Point, ScaleConfig, float]]:
        """Fallback for gauge layouts the pose model cannot recognize (e.g.
        the min number printed at the left instead of the bottom-left, so
        the max scale-end box is never detected): OCR the whole dial face,
        take the lowest and highest numbers as the scale ends, and
        interpolate the pointer angle between them directly.

        Returns (start, end, scale, value): the image positions of the min
        and max numbers, the value range they define, and the interpolated
        reading. None when the scale cannot be read reliably."""
        self.last_debug = {}
        if not self.config.enabled:
            return None
        try:
            items = self._ocr_items_with_boxes(roi)
        except (ImportError, RuntimeError, ValueError) as exc:
            if not self._engine_failed:
                self._engine_failed = True
                warnings.warn("OCR scale detection unavailable: {}".format(exc))
            return None
        tokens = []
        for text, score, x, y in items:
            if score is None or score < self._MIN_END_SCORE:
                continue
            number = parse_number(text)
            if number is None or not math.isfinite(number):
                continue
            tokens.append((number, float(x), float(y), float(score)))
        # Native and upscaled reads report the same numbers twice; keep the
        # most confident instance of each value.
        tokens = sorted(
            tokens, key=lambda item: item[3], reverse=True
        )
        unique: "List[Tuple[float, float, float]]" = []
        seen_values = set()
        for number, x, y, score in tokens:
            if number in seen_values:
                continue
            seen_values.add(number)
            unique.append((number, x, y))
        tokens = unique
        # The dial face can carry serial numbers (e.g. "80723048") alongside
        # the scale numbers; drop tokens that are orders of magnitude larger
        # than the typical scale number. The median is a stable "typical"
        # magnitude even when a serial is present.
        if len(tokens) >= 3:
            median = sorted(token[0] for token in tokens)[len(tokens) // 2]
            magnitude_limit = 100.0 * max(1.0, abs(median))
            tokens = [
                token for token in tokens if abs(token[0]) <= magnitude_limit
            ]
        # Scale numbers sit on the number ring around the dial; center
        # labels (e.g. "0.01mm" on a dial indicator) parse as numbers too
        # but sit close to the pivot. Drop tokens much closer to the center
        # than the typical number, or they can win the min/max selection
        # and wreck the angle baseline.
        if len(tokens) >= 3:
            center_x0, center_y0 = center
            distances = sorted(
                math.hypot(token[1] - center_x0, token[2] - center_y0)
                for token in tokens
            )
            ring_distance = distances[len(distances) // 2]
            tokens = [
                token
                for token in tokens
                if math.hypot(token[1] - center_x0, token[2] - center_y0)
                >= 0.6 * ring_distance
            ]
        self.last_debug["tokens"] = tokens
        if len(tokens) < 2:
            return None
        minimum = min(tokens, key=lambda item: item[0])
        maximum = max(tokens, key=lambda item: item[0])
        if minimum[0] >= maximum[0]:
            return None

        center_x, center_y = center
        start_angle = math.atan2(center_y - minimum[2], minimum[1] - center_x)
        end_angle = math.atan2(center_y - maximum[2], maximum[1] - center_x)
        pointer_angle = math.atan2(center_y - pointer[1], pointer[0] - center_x)
        sweep_cw = (start_angle - end_angle) % (2.0 * math.pi)
        sweep_ccw = (end_angle - start_angle) % (2.0 * math.pi)
        # The scale runs from the min number to the max number; the other
        # numbers on the dial all lie on that arc, so let them vote for the
        # direction. The "over the top" heuristic below only kicks in when
        # there are no intermediate numbers (min/max adjacent), e.g. on a
        # full-circle dial whose 0/100 gap is at the top the +90 degree
        # direction falls inside the gap, so the top heuristic alone picks
        # the wrong (short) arc.
        cw_count = sum(
            1
            for token in tokens
            if minimum[0] < token[0] < maximum[0]
            and (start_angle - math.atan2(center_y - token[2], token[1] - center_x))
            % (2.0 * math.pi) < sweep_cw
        )
        ccw_count = sum(
            1 for token in tokens if minimum[0] < token[0] < maximum[0]
        ) - cw_count
        if cw_count != ccw_count:
            direction = 1 if cw_count > ccw_count else -1
        else:
            top_cw = (start_angle - math.pi / 2.0) % (2.0 * math.pi)
            top_ccw = (math.pi / 2.0 - start_angle) % (2.0 * math.pi)
            if top_cw < sweep_cw:
                direction = 1  # clockwise (in the atan2 frame)
            elif top_ccw < sweep_ccw:
                direction = -1  # counter-clockwise
            else:
                return None
        if not self._MIN_ARC <= (sweep_cw if direction > 0 else sweep_ccw) <= self._MAX_ARC:
            return None

        def sweep_of(angle: float) -> float:
            if direction > 0:
                return (start_angle - angle) % (2.0 * math.pi)
            return (angle - start_angle) % (2.0 * math.pi)

        pointer_sweep = sweep_of(pointer_angle)
        # Full-circle dials (e.g. dial indicators): the numbers 10..90 span
        # nearly the whole circle and the missing 0/100 sit together in the
        # small gap at the top of the dial. Detect that gap (short AND
        # containing the +90 degree direction) and infer the two end
        # numbers at its midpoint instead of treating the min/max numbers
        # as the scale ends. Half-circle gauges (0 at the bottom-left) have
        # their gap at the bottom, so they are not affected.
        sweep = sweep_cw if direction > 0 else sweep_ccw
        gap = (2.0 * math.pi) - sweep
        end_angle = math.atan2(center_y - maximum[2], maximum[1] - center_x)
        if direction > 0:
            top_in_gap = (end_angle - math.pi / 2.0) % (2.0 * math.pi) < gap
        else:
            top_in_gap = (math.pi / 2.0 - end_angle) % (2.0 * math.pi) < gap
        if gap < 2.0 * math.pi / 3.0 and top_in_gap:
            step = (maximum[0] - minimum[0]) / max(len(tokens) - 1, 1)
            full_min = minimum[0] - step
            full_max = maximum[0] + step
            gap_mid_sweep = (sweep + gap / 2.0) % (2.0 * math.pi)
            ratio = (pointer_sweep - gap_mid_sweep) % (2.0 * math.pi) / (
                2.0 * math.pi
            )
            value = full_min + ratio * (full_max - full_min)
            gap_mid_angle = end_angle - gap / 2.0 if direction > 0 else end_angle + gap / 2.0
            ring_radius = max(
                math.hypot(token[1] - center_x, token[2] - center_y)
                for token in tokens
            )
            gap_mid_point = (
                center_x + ring_radius * math.cos(gap_mid_angle),
                center_y - ring_radius * math.sin(gap_mid_angle),
            )
            scale = self._build_scale(full_min, full_max, fallback)
            self.last_debug["scale_ends"] = {
                "start": [gap_mid_point[0], gap_mid_point[1]],
                "end": [gap_mid_point[0], gap_mid_point[1]],
                "value": value,
                "full_circle": True,
            }
            return gap_mid_point, gap_mid_point, scale, value
        # Dial numbers are rarely evenly spaced (steps shrink toward the
        # middle of the arc), so interpolate within the segment between the
        # two numbers flanking the pointer; fall back to a straight ratio
        # over the whole arc when the pointer is not between two numbers.
        ordered = sorted(tokens, key=lambda item: sweep_of(
            math.atan2(center_y - item[2], item[1] - center_x)
        ))
        chain = []
        for number, x, y in ordered:
            if not chain or number > chain[-1][0]:
                chain.append((number, x, y))
        value = None
        if chain[0][0] == minimum[0] and chain[-1][0] == maximum[0]:
            for index in range(len(chain) - 1):
                lower, upper = chain[index], chain[index + 1]
                lower_sweep = sweep_of(
                    math.atan2(center_y - lower[2], lower[1] - center_x)
                )
                upper_sweep = sweep_of(
                    math.atan2(center_y - upper[2], upper[1] - center_x)
                )
                if lower_sweep <= pointer_sweep <= upper_sweep:
                    fraction = (pointer_sweep - lower_sweep) / (
                        upper_sweep - lower_sweep
                    )
                    value = lower[0] + fraction * (upper[0] - lower[0])
                    break
        if value is None:
            ratio = min(1.0, max(0.0, pointer_sweep / sweep))
            value = minimum[0] + ratio * (maximum[0] - minimum[0])

        scale = self._build_scale(minimum[0], maximum[0], fallback)
        self.last_debug["scale_ends"] = {
            "start": [minimum[1], minimum[2]],
            "end": [maximum[1], maximum[2]],
            "value": value,
        }
        return (minimum[1], minimum[2]), (maximum[1], maximum[2]), scale, value
