import math

from .config import ScaleConfig
from .types import MeterPoints


def angle_ratio(points: MeterPoints) -> float:
    start_x, start_y = points.start
    end_x, end_y = points.end
    pointer_x, pointer_y = points.pointer
    center_x, center_y = points.center

    beginning_x_angle = math.atan2(center_y - start_y, start_x - center_x)
    end_x_angle = math.atan2(center_y - end_y, end_x - center_x)
    beginning_end_angle = 2.0 * math.pi - (end_x_angle - beginning_x_angle)
    if abs(beginning_end_angle) < 1e-12:
        raise ValueError("The scale start and end points produce a zero angle")

    pointer_x_angle = math.atan2(center_y - pointer_y, pointer_x - center_x)
    if pointer_y > center_y and pointer_x < center_x:
        beginning_pointer_angle = pointer_x_angle - beginning_x_angle
    else:
        beginning_pointer_angle = 2.0 * math.pi - (
            pointer_x_angle - beginning_x_angle
        )
    return beginning_pointer_angle / beginning_end_angle


def scale_value(points: MeterPoints, config: ScaleConfig) -> float:
    value = (config.end - config.beginning) * angle_ratio(points) + config.beginning
    return abs(value)


def compensated_value(value: float, config: ScaleConfig, enabled: bool = True) -> float:
    if not enabled:
        return value
    range_size = config.end - config.beginning
    if range_size <= 0.0:
        return value
    # The split and the compensation amounts are fractions of the scale
    # range, so they stay consistent for any (possibly OCR-detected) range.
    split = config.beginning + config.compensation_split * range_size
    if value <= split:
        return value + config.lower_compensation * range_size
    return value + config.upper_compensation * range_size
