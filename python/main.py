import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2

from meter_reader import MeterReader, MeterReaderConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--gpu", action="store_true", help="Enable NCNN Vulkan inference")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-compensation", action="store_true")
    parser.add_argument("--no-ocr-scale", action="store_true",
                        help="Disable OCR scale-range detection; use --scale-min/--scale-max")
    parser.add_argument("--ocr-padding", type=float, default=0.5,
                        help="Expand the OCR crop around the scale-end boxes by this fraction of box size")
    parser.add_argument("--scale-min", type=float, default=0.0)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--unit", default="MPa")
    parser.add_argument(
        "--aspect-ratio-threshold",
        type=float,
        default=0.0,
        help="Require meter ROI width/height to be within t and 1/t; 0 disables it",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NanoDet + YOLOv8-Pose meter reader")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    single = subparsers.add_parser("single", help="Process one image")
    single.add_argument("path", type=Path)
    add_common_arguments(single)

    folder = subparsers.add_parser("folder", help="Process images in a folder")
    folder.add_argument("path", type=Path)
    folder.add_argument(
        "--extensions", nargs="+", default=(".jpg", ".jpeg", ".png", ".bmp")
    )
    add_common_arguments(folder)

    camera = subparsers.add_parser("camera", help="Read frames from a camera")
    camera.add_argument("--device", type=int, default=0)
    add_common_arguments(camera)
    return parser


def create_reader(args: argparse.Namespace) -> MeterReader:
    config = MeterReaderConfig(
        weights_dir=args.weights.resolve(),
        use_gpu=args.gpu,
        num_threads=args.threads,
    )
    config.scale.beginning = args.scale_min
    config.scale.end = args.scale_max
    config.scale.unit = args.unit
    if not 0.0 <= args.aspect_ratio_threshold < 1.0:
        raise ValueError("--aspect-ratio-threshold must be in [0, 1)")
    config.aspect_ratio_threshold = args.aspect_ratio_threshold
    config.ocr.enabled = not args.no_ocr_scale
    config.ocr.crop_padding = args.ocr_padding
    return MeterReader(config)


def print_result(path: str, elapsed_ms: float, readings, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "source": path,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "readings": [reading.as_dict() for reading in readings],
                },
                ensure_ascii=False,
            )
        )
        return
    if not readings:
        print("{}: no readable meter ({:.2f} ms)".format(path, elapsed_ms))
        return
    for index, reading in enumerate(readings):
        print(
            "{} meter[{}]: {:.3f} {} ({:.2f} ms)".format(
                path, index, reading.display_value, reading.unit, elapsed_ms
            )
        )


def process_image(
    reader: MeterReader,
    image,
    source: str,
    args: argparse.Namespace,
    save_name: Optional[str] = None,
):
    started = time.perf_counter()
    readings = reader.read(
        image,
        apply_compensation=not args.no_compensation,
        debug=args.debug,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print_result(source, elapsed_ms, readings, args.as_json)
    rendered = None
    if args.show or args.save_dir is not None:
        rendered = reader.visualize(image, readings)
    if args.save_dir is not None and rendered is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_dir / (save_name or "result.jpg")), rendered)
    return rendered


def image_paths(folder: Path, extensions: Iterable[str]):
    allowed = {extension.lower() for extension in extensions}
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in allowed
    )


def run_single(reader: MeterReader, args: argparse.Namespace) -> int:
    image = cv2.imread(str(args.path))
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(args.path))
    rendered = process_image(reader, image, str(args.path), args, args.path.name)
    if args.show and rendered is not None:
        cv2.imshow("meter-reader", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_folder(reader: MeterReader, args: argparse.Namespace) -> int:
    paths = image_paths(args.path, args.extensions)
    if not paths:
        raise FileNotFoundError("No supported images found in {}".format(args.path))
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print("Skipping unreadable image: {}".format(path), file=sys.stderr)
            continue
        rendered = process_image(reader, image, str(path), args, path.name)
        if args.show and rendered is not None:
            cv2.imshow("meter-reader", rendered)
            if cv2.waitKey(0) & 0xFF in (27, ord("q")):
                break
    if args.show:
        cv2.destroyAllWindows()
    return 0


def run_camera(reader: MeterReader, args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera {}".format(args.device))
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            save_name = "frame_{:06d}.jpg".format(frame_index)
            rendered = process_image(
                reader, frame, "camera:{}".format(args.device), args, save_name
            )
            frame_index += 1
            if args.show and rendered is not None:
                cv2.imshow("meter-reader", rendered)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        reader = create_reader(args)
        if args.mode == "single":
            return run_single(reader, args)
        if args.mode == "folder":
            return run_folder(reader, args)
        return run_camera(reader, args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
