from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class NanoDetConfig:
    input_size: Tuple[int, int] = (320, 320)
    score_threshold: float = 0.3
    nms_threshold: float = 0.3
    num_classes: int = 1
    reg_max: int = 7
    strides: Tuple[int, ...] = (8, 16, 32, 64)
    mean: Tuple[float, float, float] = (103.53, 116.28, 123.675)
    norm: Tuple[float, float, float] = (0.017429, 0.017507, 0.017125)
    input_name: str = "data"
    output_name: str = "output"


@dataclass
class PoseConfig:
    target_size: int = 320
    max_stride: int = 32
    probability_threshold: float = 0.25
    nms_threshold: float = 0.45
    keypoint_threshold: float = 0.5
    reg_max: int = 16
    num_classes: int = 3
    keypoint_count: int = 2
    input_name: str = "images"
    output_names: Tuple[str, str, str] = ("output0", "378", "403")
    output_strides: Tuple[int, int, int] = (8, 16, 32)


@dataclass
class PointerConfig:
    binary_threshold: int = 210
    hough_threshold: int = 50
    min_line_length: int = 100
    max_line_gap: int = 150


@dataclass
class ScaleConfig:
    beginning: float = 0.0
    end: float = 1.0
    unit: str = "MPa"
    compensation_split: float = 0.5
    lower_compensation: float = 0.012
    upper_compensation: float = 0.008


@dataclass
class OcrConfig:
    enabled: bool = True
    # Expand the crop around each scale-end box by this fraction of the box
    # size on every side. Tune when the printed numbers fall outside the box.
    crop_padding: float = 0.5
    # Upscale the crop before OCR; small digits on 320-input ROIs benefit.
    upscale_factor: float = 2.0
    min_crop_size: int = 16
    use_cache: bool = True
    # Cache key quantizes the meter boxes by this many pixels so that camera
    # jitter does not defeat the cache.
    cache_quantize: int = 8
    cache_max_entries: int = 16


@dataclass
class MeterReaderConfig:
    weights_dir: Path
    use_gpu: bool = False
    num_threads: int = 4
    # The C++ example rejected every ROI outside 0.90-1.11. Keep this disabled
    # by default because real detector boxes and photographed gauges are often
    # not perfectly square.
    aspect_ratio_threshold: float = 0.0
    # When the pose model cannot find all three keypoint classes inside the
    # detector's tight box (e.g. the box cuts off one scale endpoint), the
    # reader retries on a crop padded by this fraction of the box size.
    # Set to 0.0 to disable the padded retry.
    roi_padding: float = 0.4
    nanodet: NanoDetConfig = field(default_factory=NanoDetConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    pointer: PointerConfig = field(default_factory=PointerConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)

    @property
    def nanodet_param(self) -> Path:
        return self.weights_dir / "nanodet.param"

    @property
    def nanodet_bin(self) -> Path:
        return self.weights_dir / "nanodet.bin"

    @property
    def pose_param(self) -> Path:
        return self.weights_dir / "yolov8s-pose-opt.param"

    @property
    def pose_bin(self) -> Path:
        return self.weights_dir / "yolov8s-pose-opt.bin"

    def validate(self) -> None:
        missing = [
            path
            for path in (
                self.nanodet_param,
                self.nanodet_bin,
                self.pose_param,
                self.pose_bin,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing model files: " + ", ".join(str(path) for path in missing)
            )
        if not 0.0 <= self.roi_padding < 1.0:
            raise ValueError("roi_padding must be in [0, 1)")
        if not 0.0 <= self.ocr.crop_padding <= 2.0:
            raise ValueError("ocr.crop_padding must be in [0, 2]")
