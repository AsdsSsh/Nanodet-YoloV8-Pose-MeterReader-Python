import math
import re
import warnings
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import OcrConfig, ScaleConfig
from .types import PoseDetection

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
    def _normalize_output(output) -> List[Tuple[str, Optional[float]]]:
        """Normalize the engine return value to a list of (text, score) pairs.
        Handles both the v1 style (([box, txt, score], elapse) or None) and
        the newer rapidocr package style (an object with .txts/.scores)."""
        if isinstance(output, tuple) and len(output) == 2:
            result = output[0]
        else:
            result = output
        if result is None:
            return []
        if hasattr(result, "txts"):
            texts = list(result.txts)
            scores = list(result.scores)
            return [
                (str(text), float(score) if score is not None else None)
                for text, score in zip(texts, scores)
            ]
        items = []
        for entry in result:
            if len(entry) < 3:
                continue
            text, score = entry[1], entry[2]
            if not isinstance(text, str):
                continue
            items.append((text, float(score) if score is not None else None))
        return items

    def _ocr_items(self, crop: np.ndarray) -> List[Tuple[str, Optional[float]]]:
        """OCR the crop at its native resolution and, when configured, at an
        upscaled resolution too. Results are merged and the caller picks the
        most confident number, so a misread caused by upscaling cannot win
        over a correct native read."""
        engine = self._ensure_engine()
        items = self._normalize_output(engine(crop))
        upscale = self.config.upscale_factor
        if upscale > 1.0 and max(crop.shape[0], crop.shape[1]) < 640:
            upscaled = cv2.resize(
                crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
            )
            items.extend(self._normalize_output(engine(upscaled)))
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
