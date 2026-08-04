from dataclasses import dataclass, field
from typing import List, Optional, Tuple


Point = Tuple[float, float]


@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float
    score: float
    label: int

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass
class PoseDetection(Detection):
    keypoints: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class MeterPoints:
    start: Point
    end: Point
    center: Point
    pointer: Point
    pointer_line: Optional[Tuple[int, int, int, int]] = None


@dataclass
class MeterReading:
    detection: Detection
    value: float
    display_value: float
    unit: str
    points: MeterPoints

    def as_dict(self) -> dict:
        return {
            "bbox": [
                round(self.detection.x, 3),
                round(self.detection.y, 3),
                round(self.detection.width, 3),
                round(self.detection.height, 3),
            ],
            "score": round(self.detection.score, 6),
            "value": round(self.value, 6),
            "display_value": round(self.display_value, 6),
            "unit": self.unit,
            "points": {
                "start": list(self.points.start),
                "end": list(self.points.end),
                "center": list(self.points.center),
                "pointer": list(self.points.pointer),
                "pointer_line": list(self.points.pointer_line)
                if self.points.pointer_line is not None
                else None,
            },
        }
