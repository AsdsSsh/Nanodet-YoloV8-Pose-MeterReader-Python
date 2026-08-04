from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


class NcnnBackend:
    """Small compatibility layer around the official ncnn Python binding."""

    def __init__(
        self,
        param_path: Path,
        bin_path: Path,
        use_gpu: bool = False,
        num_threads: int = 4,
    ) -> None:
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError(
                "The 'ncnn' Python package is required. Install python/requirements.txt."
            ) from exc

        self.ncnn = ncnn
        self.net = ncnn.Net()
        get_gpu_count = getattr(ncnn, "get_gpu_count", lambda: 0)
        self.net.opt.use_vulkan_compute = bool(use_gpu and get_gpu_count() > 0)
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.num_threads = int(num_threads)

        param_result = self.net.load_param(str(param_path))
        model_result = self.net.load_model(str(bin_path))
        if param_result not in (None, 0) or model_result not in (None, 0):
            raise RuntimeError(
                "Failed to load NCNN model: {} / {}".format(param_path, bin_path)
            )

    def from_pixels(
        self,
        image: np.ndarray,
        pixel_type_name: str,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Any:
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        pixel_type = getattr(self.ncnn.Mat.PixelType, pixel_type_name)
        if target_width is None or target_height is None:
            return self.ncnn.Mat.from_pixels(image, pixel_type, width, height)
        return self.ncnn.Mat.from_pixels_resize(
            image,
            pixel_type,
            width,
            height,
            int(target_width),
            int(target_height),
        )

    @staticmethod
    def normalize(mat: Any, mean: Optional[Sequence[float]], norm: Sequence[float]) -> None:
        mean_values = None if mean is None else list(mean)
        mat.substract_mean_normalize(mean_values, list(norm))

    def infer(self, input_name: str, input_mat: Any, output_names: Sequence[str]):
        extractor = self.net.create_extractor()
        extractor.input(input_name, input_mat)
        outputs = []
        for name in output_names:
            result = extractor.extract(name)
            if isinstance(result, tuple):
                return_code, output = result
                if return_code != 0:
                    raise RuntimeError("NCNN failed to extract output '{}'".format(name))
            else:
                output = result
            outputs.append(np.asarray(output, dtype=np.float32).copy())
        return outputs
