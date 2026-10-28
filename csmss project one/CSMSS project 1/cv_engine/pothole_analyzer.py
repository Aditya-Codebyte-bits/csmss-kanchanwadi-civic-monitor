"""
Scientifically Grounded Pothole & Road Surface Anomaly Analyzer (`cv_engine/pothole_analyzer.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering.

Pipeline stages:
1. Convert road ROI to grayscale & HSV color space.
2. Apply Gaussian smoothing + Contrast Limited Adaptive Histogram Equalization (CLAHE).
3. Perform morphological Black-Hat transform and morphological closing to isolate dark road cavities.
4. Combine Canny edge map + adaptive darkness thresholding + contour geometry filtering.
5. Compute calibrated physical area (`surface_area_sqm = pixel_area / (px_per_meter ** 2)`),
   defensible 0-100 Road Damage Severity Index (50% normalized area + 30% interior shadow
   darkness + 20% perimeter-to-area jaggedness), heuristic shadow-intensity depth proxy (cm),
   and estimated civil repair cost in INR.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    BASE_REPAIR_RATE_INR_PER_SQM,
    CAMPUS_ZONES,
    MIN_REPAIR_MOBILIZATION_COST_INR,
    PIXELS_PER_METER_BY_ZONE,
)


def classify_pothole_severity(severity_index: float, surface_area_sqm: float) -> Tuple[str, str]:
    """
    Map the 0-100 Road Damage Severity Index and calibrated area (sqm) to:
    - `pothole_class`: `'Surface Crack'`, `'Moderate Cavity'`, or `'Critical Hazard'`
    - `incident_severity`: `'Low'`, `'Medium'`, `'High'`, or `'Critical'`
    """
    if severity_index >= 72.0 or surface_area_sqm >= 1.15:
        return "Critical Hazard", "Critical"
    if severity_index >= 42.0 or surface_area_sqm >= 0.45:
        return "Moderate Cavity", "High" if severity_index >= 58.0 else "Medium"
    return "Surface Crack", "Medium" if severity_index >= 25.0 else "Low"


def analyze_road_surface(
    frame_bgr: np.ndarray,
    zone_id: str = "ZONE-02",
    roi_mask: Optional[np.ndarray] = None,
    px_per_meter: Optional[float] = None,
    clahe_clip_limit: float = 2.5,
    min_cavity_area_px: int = 260,
    vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Dict[str, Any]:
    """
    Analyze road ROI for potholes, asphalt ravelling, and structural surface cavities.

    Formula for Road Damage Severity Index (0–100):
        `severity_index = 100 * (0.50 * norm_area + 0.30 * shadow_darkness_ratio + 0.20 * jaggedness_score)`

    Returns:
        Dictionary containing detected cavity contours, aggregate and primary cavity
        telemetry (`surface_area_sqm`, `severity_index`, `estimated_depth_proxy_cm`,
        `pothole_class`, `estimated_repair_cost_inr`), and intermediate visual maps
        (`clahe_gray`, `canny_edges`, `blackhat_map`, `anomaly_heatmap_rgb`) for Stage 2 X-Ray.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Input frame_bgr is empty")

    h, w = frame_bgr.shape[:2]
    zone_meta = CAMPUS_ZONES.get(zone_id, CAMPUS_ZONES["ZONE-02"])
    ppm = float(
        px_per_meter
        if px_per_meter is not None
        else PIXELS_PER_METER_BY_ZONE.get(zone_id, zone_meta.get("px_per_meter", 84.5))
    )
    crit_weight = float(zone_meta.get("criticality_weight", 1.2))

    # Build effective road mask (restrict to ROI polygon and exclude detected vehicle boxes)
    if roi_mask is not None and roi_mask.shape[:2] == (h, w):
        effective_mask = roi_mask.copy()
    else:
        effective_mask = np.zeros((h, w), dtype=np.uint8)
        effective_mask[int(h * 0.30) : int(h * 0.94), int(w * 0.08) : int(w * 0.92)] = 255

    if vehicle_boxes:
        for (vx1, vy1, vx2, vy2) in vehicle_boxes:
            cv2.rectangle(effective_mask, (int(vx1), int(vy1)), (int(vx2), int(vy2)), 0, -1)

    # 1. Convert to Grayscale & HSV
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # 2. Gaussian Blur + CLAHE Contrast Enhancement
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clip_val = max(0.5, float(clahe_clip_limit))
    clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(blurred)

    # 3. Morphological Black-Hat transform to isolate dark cavities against lighter asphalt
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    blackhat = cv2.morphologyEx(clahe_gray, cv2.MORPH_BLACKHAT, kernel_bh)

    # Combine darkness depression with Black-Hat response inside road ROI
    road_pixels = v_channel[effective_mask > 0]
    median_road_v = float(np.median(road_pixels)) if road_pixels.size > 0 else 120.0
    dark_depression = np.clip(median_road_v - v_channel.astype(np.float32), 0, 255).astype(np.uint8)

    # Suppress high-saturation painted markings (yellow/blue/red) so road stripes aren't mistaken for cavities
    low_sat_mask = (s_channel < 115).astype(np.uint8) * 255
    valid_road_mask = cv2.bitwise_and(effective_mask, low_sat_mask)

    combined_response = cv2.addWeighted(blackhat, 0.55, dark_depression, 0.75, 0)
    combined_response = cv2.bitwise_and(combined_response, valid_road_mask)

    # 4. Canny Edge Map + Morphological Closing for cavity boundary segmentation
    canny_edges = cv2.Canny(clahe_gray, 55, 145)
    canny_edges = cv2.bitwise_and(canny_edges, valid_road_mask)

    _, thresh_mask = cv2.threshold(combined_response, 36, 255, cv2.THRESH_BINARY)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours_info = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    cavities: List[Dict[str, Any]] = []
    kept_contours: List[np.ndarray] = []

    for cnt in contours:
        pixel_area = float(cv2.contourArea(cnt))
        if pixel_area < float(min_cavity_area_px):
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = float(cw) / float(max(1, ch))
        # Reject ultra-thin line artifacts
        if aspect_ratio > 8.5 or aspect_ratio < 0.12:
            continue

        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0.0:
            continue

        # Compute interior shadow darkness ratio in [0.0, 1.0]
        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        interior_v = v_channel[cnt_mask > 0]
        if interior_v.size == 0:
            continue

        mean_interior_v = float(np.mean(interior_v))
        p15_interior_v = float(np.percentile(interior_v, 15))
        # Relative shadow drop compared to surrounding road median brightness
        relative_drop = max(0.0, (median_road_v - p15_interior_v) / max(40.0, median_road_v))
        absolute_darkness = max(0.0, (255.0 - mean_interior_v) / 255.0)
        shadow_darkness_ratio = float(np.clip(0.65 * relative_drop + 0.35 * absolute_darkness, 0.05, 1.0))

        # reject bright patches that aren't true cavities
        if shadow_darkness_ratio < 0.22:
            continue

        # Physical surface area in square meters via homography scale factor
        surface_area_sqm = round(pixel_area / (ppm ** 2), 4)

        # Normalized area component (0.0 to 1.0, saturating near 1.50 sqm)
        norm_area = float(np.clip(surface_area_sqm / 1.45, 0.04, 1.0))

        # Edge jaggedness / perimeter-to-area ratio component (isoperimetric quotient inverse)
        # For a perfect circle, P^2 / (4 * pi * A) == 1.0; jagged potholes have higher values (1.4 - 4.0)
        shape_complexity = (perimeter ** 2) / max(1.0, 4.0 * np.pi * pixel_area)
        jaggedness_score = float(np.clip((shape_complexity - 1.0) / 2.6, 0.10, 1.0))

        # Composite 0-100 Road Damage Severity Index (50% Area + 30% Shadow Darkness + 20% Jaggedness)
        severity_index = round(
            float(
                np.clip(
                    100.0
                    * (
                        0.50 * norm_area
                        + 0.30 * shadow_darkness_ratio
                        + 0.20 * jaggedness_score
                    ),
                    5.0,
                    99.5,
                )
            ),
            2,
        )

        # Heuristic Shadow-Intensity Proxy for Cavity Depth (cm)
        estimated_depth_proxy_cm = round(
            float(
                np.clip(
                    1.2 + (shadow_darkness_ratio * 9.5) + (np.sqrt(surface_area_sqm) * 3.4),
                    0.5,
                    28.0,
                )
            ),
            2,
        )

        pothole_class, inc_severity = classify_pothole_severity(severity_index, surface_area_sqm)
        severity_multiplier = 1.0 + (severity_index / 100.0) * 0.85 * crit_weight
        estimated_repair_cost_inr = round(
            MIN_REPAIR_MOBILIZATION_COST_INR
            + surface_area_sqm * BASE_REPAIR_RATE_INR_PER_SQM * severity_multiplier,
            2,
        )

        confidence_score = round(
            float(np.clip(0.68 + 0.28 * shadow_darkness_ratio, 0.65, 0.98)), 3
        )

        kept_contours.append(cnt)
        cavities.append(
            {
                "bbox": (int(x), int(y), int(x + cw), int(y + ch)),
                "pixel_area": round(pixel_area, 1),
                "perimeter_px": round(perimeter, 1),
                "surface_area_sqm": surface_area_sqm,
                "norm_area_score": round(norm_area, 4),
                "shadow_darkness_ratio": round(shadow_darkness_ratio, 4),
                "jaggedness_score": round(jaggedness_score, 4),
                "severity_index": severity_index,
                "estimated_depth_proxy_cm": estimated_depth_proxy_cm,
                "depth_estimation_method": "Heuristic Shadow-Intensity Proxy (Monocular CV)",
                "pothole_class": pothole_class,
                "severity": inc_severity,
                "estimated_repair_cost_inr": estimated_repair_cost_inr,
                "confidence_score": confidence_score,
                "contour": cnt,
            }
        )

    cavities.sort(key=lambda c: c["severity_index"], reverse=True)

    # Build Stage 2 Preprocessed X-Ray image (High-contrast CLAHE + Canny edge + HSV anomaly heatmap)
    clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
    heatmap_bgr = cv2.applyColorMap(combined_response, cv2.COLORMAP_TURBO)
    blended_stage2 = cv2.addWeighted(clahe_bgr, 0.68, heatmap_bgr, 0.32, 0)
    # Highlight Canny edges in bright cyan
    blended_stage2[canny_edges > 0] = (255, 235, 0)
    stage2_preprocessed_rgb = cv2.cvtColor(blended_stage2, cv2.COLOR_BGR2RGB)

    if cavities:
        total_area_sqm = round(sum(c["surface_area_sqm"] for c in cavities), 4)
        primary = cavities[0]
        # Aggregate severity index weighted by primary cavity and total damaged area
        agg_norm_area = float(np.clip(total_area_sqm / 1.45, 0.04, 1.0))
        agg_severity_index = round(
            float(
                np.clip(
                    max(
                        primary["severity_index"],
                        100.0
                        * (
                            0.50 * agg_norm_area
                            + 0.30 * primary["shadow_darkness_ratio"]
                            + 0.20 * primary["jaggedness_score"]
                        ),
                    ),
                    5.0,
                    99.5,
                )
            ),
            2,
        )
        agg_class, agg_inc_severity = classify_pothole_severity(agg_severity_index, total_area_sqm)
        agg_multiplier = 1.0 + (agg_severity_index / 100.0) * 0.85 * crit_weight
        agg_repair_cost_inr = round(
            MIN_REPAIR_MOBILIZATION_COST_INR
            + total_area_sqm * BASE_REPAIR_RATE_INR_PER_SQM * agg_multiplier,
            2,
        )
        summary_telemetry: Optional[Dict[str, Any]] = {
            "detected": True,
            "cavity_count": len(cavities),
            "surface_area_sqm": total_area_sqm,
            "severity_index": agg_severity_index,
            "estimated_depth_proxy_cm": primary["estimated_depth_proxy_cm"],
            "depth_estimation_method": "Heuristic Shadow-Intensity Proxy (Monocular CV)",
            "pothole_class": agg_class,
            "severity": agg_inc_severity,
            "estimated_repair_cost_inr": agg_repair_cost_inr,
            "confidence_score": primary["confidence_score"],
            "primary_bbox": primary["bbox"],
        }
    else:
        summary_telemetry = {
            "detected": False,
            "cavity_count": 0,
            "surface_area_sqm": 0.0,
            "severity_index": 0.0,
            "estimated_depth_proxy_cm": 0.0,
            "depth_estimation_method": "Heuristic Shadow-Intensity Proxy (Monocular CV)",
            "pothole_class": "Surface Crack",
            "severity": "Low",
            "estimated_repair_cost_inr": 0.0,
            "confidence_score": 0.0,
            "primary_bbox": None,
        }

    return {
        "zone_id": zone_id,
        "px_per_meter": ppm,
        "clahe_clip_limit": clip_val,
        "cavities": cavities,
        "contours": kept_contours,
        "cavity_mask": cleaned_mask,
        "summary": summary_telemetry,
        "stage2_preprocessed_rgb": stage2_preprocessed_rgb,
    }