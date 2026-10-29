"""
Polygon ROI Parking & Lane Obstruction Geometry Engine (`cv_engine/parking_roi.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering.

Computes exact pixel-level and polygon-clipped intersection between detected
vehicle bounding boxes and campus zone ROI polygons using OpenCV `cv2.fillPoly`
bitwise masks and Sutherland-Hodgman polygon clipping.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from config.settings import CAMPUS_ZONES, ROI_POLYGONS_BY_ZONE


def parse_roi_coordinates(
    roi_source: Union[str, Sequence[Sequence[float]], None],
    zone_id: str = "ZONE-01",
) -> List[Tuple[float, float]]:
    """
    Parse normalized ROI polygon coordinates from either a JSON string
    (`roi_coordinates_json` from SQLite `campus_zones`) or a list of `(x, y)` pairs.
    """
    if roi_source is None:
        return list(ROI_POLYGONS_BY_ZONE.get(zone_id, ROI_POLYGONS_BY_ZONE["ZONE-01"]))

    if isinstance(roi_source, str):
        try:
            loaded = json.loads(roi_source)
            return [(float(pt[0]), float(pt[1])) for pt in loaded]
        except Exception:
            return list(ROI_POLYGONS_BY_ZONE.get(zone_id, ROI_POLYGONS_BY_ZONE["ZONE-01"]))

    return [(float(pt[0]), float(pt[1])) for pt in roi_source]


def denormalize_roi_polygon(
    normalized_points: Sequence[Sequence[float]],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Convert normalized `[(x_norm, y_norm), ...]` coordinates (or already-pixel coordinates)
    into integer pixel coordinates of shape `(N, 2)` with dtype `np.int32`.
    """
    pts: List[List[int]] = []
    for pt in normalized_points:
        x_val, y_val = float(pt[0]), float(pt[1])
        if x_val <= 1.05 and y_val <= 1.05:
            px = int(round(np.clip(x_val, 0.0, 1.0) * frame_width))
            py = int(round(np.clip(y_val, 0.0, 1.0) * frame_height))
        else:
            px = int(round(np.clip(x_val, 0.0, float(frame_width))))
            py = int(round(np.clip(y_val, 0.0, float(frame_height))))
        pts.append([px, py])
    return np.array(pts, dtype=np.int32)


def build_roi_mask(
    frame_shape: Tuple[int, ...],
    roi_polygon_px: np.ndarray,
) -> np.ndarray:
    """Render a binary uint8 mask (`0` or `255`) of the zone ROI polygon using `cv2.fillPoly`."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if roi_polygon_px is not None and len(roi_polygon_px) >= 3:
        cv2.fillPoly(mask, [roi_polygon_px.astype(np.int32)], 255)
    return mask


def compute_box_roi_intersection(
    bbox: Tuple[int, int, int, int],
    roi_polygon_px: np.ndarray,
    frame_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Compute exact intersection between a vehicle bounding box `(x1, y1, x2, y2)`
    and the campus zone ROI polygon using `cv2.fillPoly` bitwise masks.

    Returns:
        - `intersection_area`: exact overlapping pixel count
        - `vehicle_box_area`: total bounding box pixel area
        - `overlap_ratio`: `intersection_area / vehicle_box_area` in `[0.0, 1.0]`
        - `intersection_mask`: binary `uint8` mask of the overlapping region
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1_c = max(0, min(w, x1))
    y1_c = max(0, min(h, y1))
    x2_c = max(0, min(w, x2))
    y2_c = max(0, min(h, y2))

    bw = max(0, x2_c - x1_c)
    bh = max(0, y2_c - y1_c)
    vehicle_box_area = float(bw * bh)

    roi_mask = build_roi_mask((h, w), roi_polygon_px)
    box_mask = np.zeros((h, w), dtype=np.uint8)
    if bw > 0 and bh > 0:
        box_poly = np.array(
            [[x1_c, y1_c], [x2_c, y1_c], [x2_c, y2_c], [x1_c, y2_c]],
            dtype=np.int32,
        )
        cv2.fillPoly(box_mask, [box_poly], 255)

    intersection_mask = cv2.bitwise_and(roi_mask, box_mask)
    intersection_area = float(cv2.countNonZero(intersection_mask))
    overlap_ratio = (
        round(min(1.0, intersection_area / vehicle_box_area), 4)
        if vehicle_box_area > 0.0
        else 0.0
    )

    return {
        "bbox": (x1_c, y1_c, x2_c, y2_c),
        "intersection_area": intersection_area,
        "vehicle_box_area": vehicle_box_area,
        "overlap_ratio": overlap_ratio,
        "intersection_mask": intersection_mask,
    }


def evaluate_parking_violations(
    vehicle_detections: Sequence[Dict[str, Any]],
    zone_id: str = "ZONE-01",
    roi_coordinates_json: Optional[Union[str, Sequence[Sequence[float]]]] = None,
    frame_shape: Tuple[int, int] = (720, 1280),
    roi_overlap_threshold: float = 0.30,
) -> Dict[str, Any]:
    """
    Evaluate all detected vehicles in a frame against the zone's ROI polygon.
    Flags an `Emergency Lane Blocked` (in `ZONE-01` Main Gate & Security Plaza)
    or `Unlawful Parking` violation when `overlap_ratio >= roi_overlap_threshold`.
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    norm_pts = parse_roi_coordinates(roi_coordinates_json, zone_id=zone_id)
    roi_polygon_px = denormalize_roi_polygon(norm_pts, frame_width=w, frame_height=h)
    roi_mask = build_roi_mask((h, w), roi_polygon_px)
    combined_intersection_mask = np.zeros((h, w), dtype=np.uint8)

    zone_meta = CAMPUS_ZONES.get(zone_id, CAMPUS_ZONES["ZONE-01"])
    crit_weight = float(zone_meta.get("criticality_weight", 1.2))

    evaluated_vehicles: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []

    for idx, det in enumerate(vehicle_detections):
        bbox = tuple(int(v) for v in det.get("bbox", (0, 0, 0, 0)))
        geom = compute_box_roi_intersection(
            bbox=bbox,
            roi_polygon_px=roi_polygon_px,
            frame_shape=(h, w),
        )
        overlap_ratio = float(geom["overlap_ratio"])
        is_violation = overlap_ratio >= float(roi_overlap_threshold)

        combined_intersection_mask = cv2.bitwise_or(
            combined_intersection_mask, geom["intersection_mask"]
        )

        # Determine violation classification and severity based on zone & overlap
        is_whitelisted = det.get("is_whitelisted", True)
        if is_violation:
            if zone_id == "ZONE-01":
                incident_type = "Emergency Lane Blocked"
            elif is_whitelisted is False or zone_id == "ZONE-04":
                incident_type = "Unauthorized Vehicle" if is_whitelisted is False else "Unlawful Parking"
            else:
                incident_type = "Unlawful Parking"

            weighted_score = overlap_ratio * crit_weight
            if incident_type == "Emergency Lane Blocked" or weighted_score >= 0.85:
                severity = "Critical"
            elif weighted_score >= 0.60 or incident_type == "Unauthorized Vehicle":
                severity = "High"
            elif weighted_score >= 0.38:
                severity = "Medium"
            else:
                severity = "Low"
        else:
            incident_type = "Compliant Transit"
            severity = "Low"

        enriched = {
            **det,
            "vehicle_index": idx,
            "bbox": geom["bbox"],
            "vehicle_box_area": geom["vehicle_box_area"],
            "intersection_area": geom["intersection_area"],
            "overlap_ratio": overlap_ratio,
            "roi_overlap_threshold": float(roi_overlap_threshold),
            "is_violation": is_violation,
            "incident_type": incident_type,
            "severity": severity,
        }
        evaluated_vehicles.append(enriched)
        if is_violation:
            violations.append(enriched)

    return {
        "zone_id": zone_id,
        "roi_polygon_px": roi_polygon_px,
        "roi_mask": roi_mask,
        "combined_intersection_mask": combined_intersection_mask,
        "roi_overlap_threshold": float(roi_overlap_threshold),
        "evaluated_vehicles": evaluated_vehicles,
        "violations": violations,
        "violation_count": len(violations),
    }