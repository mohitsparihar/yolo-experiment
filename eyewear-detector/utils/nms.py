"""Non-maximum suppression for overlapping bounding boxes."""

import numpy as np


def nms(boxes: list[list[float]], scores: list[float], iou_threshold: float = 0.5) -> list[int]:
    """
    Apply non-maximum suppression to remove overlapping boxes.

    Args:
        boxes: List of [x1, y1, x2, y2] bounding boxes (normalized or pixel coords).
        scores: Confidence scores for each box.
        iou_threshold: IoU threshold above which boxes are suppressed.

    Returns:
        List of indices to keep.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep
