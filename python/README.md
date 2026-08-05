# Python NCNN Meter Reader

This directory is a Python implementation of the original C++ inference
pipeline. It uses the checked-in NCNN model files in '../weights'; no model
conversion is required.

The pipeline is:

~~~text
image -> NanoDet meter box -> cropped meter ROI -> YOLOv8-Pose keypoints
      -> OpenCV pointer-line refinement -> angular scale calculation
~~~

The defaults intentionally match the C++ program:

- NanoDet input: 320 x 320, score threshold 0.3, NMS threshold 0.3
- YOLOv8-Pose input long side: 320, score threshold 0.25, NMS threshold 0.45
- Scale range: 0.0 to 1.0 MPa
- Empirical compensation: +1.2% of the scale range at or below the midpoint,
  otherwise +0.8% of the scale range. Proportional to the range so it stays
  consistent when OCR detects a different scale
- ROI aspect-ratio filter: disabled by default. The C++ threshold of 0.90 was
  too restrictive for common photographed gauges.

## Setup

Use Python 3.10 or newer if possible. The machine's system Python is 3.8,
which is end-of-life and may not receive current NCNN wheels.

~~~powershell
cd python
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

'opencv-contrib-python' provides the optimized thinning operation used for
pointer extraction. The implementation has a built-in fallback if the
contrib module is unavailable.

### OCR scale-range detection (optional)

The meter scale range is normally fixed (0.0-1.0 MPa) in the configuration.
To read the range dynamically from the dial instead, install the OCR engine:

~~~powershell
python -m pip install --no-deps -r requirements-ocr.txt
~~~

'rapidocr-onnxruntime' must be installed with '--no-deps': its own
dependencies would pull in 'opencv-python', which conflicts with
'opencv-contrib-python' and can break the thinning operation. Its runtime
dependencies are already listed in 'requirements.txt'.

OCR is enabled by default. It crops the two scale-end boxes detected by the
pose model (left_rect/right_rect), recognizes the numbers next to them, and
uses the detected range for the reading. When OCR fails or returns an
unreliable range, the reader silently falls back to the configured static
scale. The scale actually used is reported in the JSON output
('scale_source' is 'ocr' or 'config', plus 'scale_begin'/'scale_end').
Results are cached per meter box so the camera mode does not re-run OCR on
every frame.

If 'pip install ncnn' has no wheel for the selected Python version, build the
official NCNN Python binding for the same interpreter, then rerun the
commands below. The rest of this project does not need to change.

## Run

Run these commands from 'python/' after activating the environment:

~~~powershell
python main.py single ..\example.jpg --show
python main.py folder ..\images --save-dir ..\outputs
python main.py camera --device 0 --show
~~~

Useful options:

~~~powershell
python main.py single ..\example.jpg --json
python main.py single ..\example.jpg --scale-min 0 --scale-max 1.6 --unit MPa
python main.py single ..\example.jpg --no-compensation
python main.py single ..\example.jpg --no-ocr-scale
python main.py single ..\example.jpg --ocr-padding 0.3
python main.py single ..\example.jpg --gpu
python main.py single ..\example.jpg --aspect-ratio-threshold 0.90
python main.py single ..\example.jpg --debug
~~~

The default model directory is '../weights'. Override it with '--weights' when
deploying model files elsewhere.

Unlike the original C++ implementation, processing does not pause for a GUI
window unless '--show' is passed. 'folder' accepts JPG, JPEG, PNG, and BMP.

## Verification

The decoder and geometry tests do not require NCNN or OpenCV:

~~~powershell
$env:PYTHONPATH = (Resolve-Path .)
python -m unittest discover -s tests -v
~~~

After dependencies are installed, perform end-to-end verification on a fixed
set of meter images. Compare the Python JSON output with the original C++
output for:

1. meter bounding boxes;
2. three pose classes and their keypoints;
3. final compensated reading.

Small numeric differences are expected because the Python implementation uses
standard exponential functions whereas the C++ NanoDet decoder uses an
approximate fast exponential.
