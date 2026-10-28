"""
4-Stage Explainable X-Ray Visualizer (`cv_engine/visualizer.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering.

Generates 4 RGB `np.ndarray` frames for every analyzed camera frame so the
Streamlit X-Ray Inspector can display step-by-step computer vision proof:
  - `stage1_raw`: Original input frame with timestamp and camera zone header.
  - `stage2_preprocessed`: High-contrast CLAHE + Canny edge / HSV anomaly heatmap overlay.
  - `stage3_geometry`: Semi-transparent ROI polygon zone overlay + vehicle footprint
    intersection mask + pothole contour mesh.
  - `stage4_hud`: Final tactical HUD frame with corner-bracket bounding boxes,
    plate callout badge, severity meter, latency telemetry, and GPS watermark.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config.settings import CAMPUS_ZONES

IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist_display() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _draw_top_banner(
    img_bgr: np.ndarray,
    left_text: str,
    right_text: str,
    accent_bgr: Tuple[int, int, int] = (215, 132, 2),
) -> None:
    """Render a sleek semi-transparent top telemetry banner on `img_bgr` in-place."""
    h, w = img_bgr.shape[:2]
    banner_h = max(34, int(h * 0.065))
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (26, 20, 12), -1)
    cv2.addWeighted(overlay, 0.82, img_bgr, 0.18, 0, img_bgr)
    cv2.line(img_bgr, (0, banner_h), (w, banner_h), accent_bgr, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.62, w / 1600.0))
    cv2.putText(
        img_bgr,
        left_text,
        (14, int(banner_h * 0.68)),
        font,
        scale,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )
    (tw, _), _ = cv2.getTextSize(right_text, font, scale, 1)
    cv2.putText(
        img_bgr,
        right_text,
        (max(14, w - tw - 14), int(banner_h * 0.68)),
        font,
        scale,
        (56, 189, 248),
        1,
        cv2.LINE_AA,
    )


def _draw_corner_bracket_box(
    img_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 18,
) -> None:
    """Draw tactical corner-bracket bounding box on `img_bgr`."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)

    cl = min(corner_len, max(6, (x2 - x1) // 3), max(6, (y2 - y1) // 3))
    # Top-left
    cv2.line(img_bgr, (x1, y1), (x1 + cl, y1), color_bgr, thickness + 1, cv2.LINE_AA)
    cv2.line(img_bgr, (x1, y1), (x1, y1 + cl), color_bgr, thickness + 1, cv2.LINE_AA)
    # Top-right
    cv2.line(img_bgr, (x2, y1), (x2 - cl, y1), color_bgr, thickness + 1, cv2.LINE_AA)
    cv2.line(img_bgr, (x2, y1), (x2, y1 + cl), color_bgr, thickness + 1, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img_bgr, (x1, y2), (x1 + cl, y2), color_bgr, thickness + 1, cv2.LINE_AA)
    cv2.line(img_bgr, (x1, y2), (x1, y2 - cl), color_bgr, thickness + 1, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img_bgr, (x2, y2), (x2 - cl, y2), color_bgr, thickness + 1, cv2.LINE_AA)
    cv2.line(img_bgr, (x2, y2), (x2, y2 - cl), color_bgr, thickness + 1, cv2.LINE_AA)


def render_stage1_raw(
    frame_bgr: np.ndarray,
    zone_id: str = "ZONE-01",
    timestamp_str: Optional[str] = None,
) -> np.ndarray:
    """
    Stage 1 (`stage1_raw`): Original input frame with timestamp and camera zone header.
    Returns RGB `np.ndarray`.
    """
    canvas = frame_bgr.copy()
    zone_meta = CAMPUS_ZONES.get(zone_id, CAMPUS_ZONES["ZONE-01"])
    zone_name = zone_meta.get("zone_name", zone_id)
    ts = timestamp_str or _now_ist_display()

    _draw_top_banner(
        canvas,
        left_text=f"STAGE 1 [RAW FEED] | {zone_id}: {zone_name}",
        right_text=ts,
        accent_bgr=(199, 132, 2),
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def render_stage2_preprocessed(
    stage2_preprocessed_rgb: np.ndarray,
    zone_id: str = "ZONE-01",
    clahe_clip_limit: float = 2.5,
) -> np.ndarray:
    """
    Stage 2 (`stage2_preprocessed`): High-contrast CLAHE + Canny edge / HSV anomaly
    heatmap overlay showing what the algorithm 'sees'.
    Returns RGB `np.ndarray`.
    """
    canvas_bgr = cv2.cvtColor(stage2_preprocessed_rgb, cv2.COLOR_RGB2BGR)
    _draw_top_banner(
        canvas_bgr,
        left_text=f"STAGE 2 [CLAHE + CANNY + HSV ANOMALY HEATMAP] | {zone_id}",
        right_text=f"CLAHE Clip: {clahe_clip_limit:.1f} | Edge + BlackHat",
        accent_bgr=(248, 189, 56),
    )
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def render_stage3_geometry(
    frame_bgr: np.ndarray,
    zone_id: str,
    roi_polygon_px: np.ndarray,
    combined_intersection_mask: np.ndarray,
    evaluated_vehicles: Sequence[Dict[str, Any]],
    pothole_cavities: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """
    Stage 3 (`stage3_geometry`): Semi-transparent ROI polygon zone overlay +
    vehicle footprint intersection mask + pothole contour mesh.
    Returns RGB `np.ndarray`.
    """
    canvas = frame_bgr.copy()
    overlay = canvas.copy()

    # 1. Semi-transparent cyan/blue ROI polygon fill
    if roi_polygon_px is not None and len(roi_polygon_px) >= 3:
        cv2.fillPoly(overlay, [roi_polygon_px.astype(np.int32)], (210, 140, 20))
        cv2.addWeighted(overlay, 0.26, canvas, 0.74, 0, canvas)
        cv2.polylines(
            canvas,
            [roi_polygon_px.astype(np.int32)],
            isClosed=True,
            color=(255, 210, 40),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # 2. Highlight vehicle-ROI footprint intersection pixels in high-visibility crimson/magenta
    if (
        combined_intersection_mask is not None
        and combined_intersection_mask.shape[:2] == canvas.shape[:2]
    ):
        intersect_overlay = canvas.copy()
        intersect_overlay[combined_intersection_mask > 0] = (50, 40, 245)
        cv2.addWeighted(intersect_overlay, 0.52, canvas, 0.48, 0, canvas)

    # 3. Draw vehicle footprint boxes & overlap ratio labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for veh in evaluated_vehicles:
        x1, y1, x2, y2 = [int(v) for v in veh["bbox"]]
        ov = float(veh.get("overlap_ratio", 0.0))
        is_viol = bool(veh.get("is_violation", False))
        col = (68, 68, 239) if is_viol else (129, 185, 16)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"ROI Overlap: {ov * 100:.1f}%",
            (x1, max(48, y1 - 8)),
            font,
            0.48,
            col,
            2,
            cv2.LINE_AA,
        )

    # 4. Draw pothole contour mesh & centroids
    for cav in pothole_cavities:
        cnt = cav.get("contour")
        if cnt is not None:
            cv2.drawContours(canvas, [cnt], -1, (0, 180, 255), 2, cv2.LINE_AA)
            bx1, by1, bx2, by2 = cav["bbox"]
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            # Draw internal mesh crosshairs across the cavity
            cv2.line(canvas, (bx1, cy), (bx2, cy), (0, 220, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx, by1), (cx, by2), (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"Cavity {cav['surface_area_sqm']:.2f}m2 | SI:{cav['severity_index']:.0f}",
                (bx1, max(48, by1 - 6)),
                font,
                0.46,
                (0, 225, 255),
                1,
                cv2.LINE_AA,
            )

    _draw_top_banner(
        canvas,
        left_text=f"STAGE 3 [POLYGON ROI GEOMETRY & CONTOUR MESH] | {zone_id}",
        right_text=f"Vehicles: {len(evaluated_vehicles)} | Cavities: {len(pothole_cavities)}",
        accent_bgr=(16, 185, 129),
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def render_stage4_hud(
    frame_bgr: np.ndarray,
    zone_id: str,
    roi_polygon_px: np.ndarray,
    evaluated_vehicles: Sequence[Dict[str, Any]],
    pothole_summary: Dict[str, Any],
    pothole_cavities: Sequence[Dict[str, Any]],
    latency_ms: Dict[str, float],
    engine_mode: str,
    timestamp_str: Optional[str] = None,
) -> np.ndarray:
    """
    Stage 4 (`stage4_hud`): Final tactical HUD frame with corner-bracket bounding boxes,
    plate callout badge, severity meter, latency telemetry, and GPS watermark.
    Returns RGB `np.ndarray`.
    """
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    zone_meta = CAMPUS_ZONES.get(zone_id, CAMPUS_ZONES["ZONE-01"])
    zone_name = zone_meta.get("zone_name", zone_id)
    lat = float(zone_meta.get("latitude", 19.8268))
    lon = float(zone_meta.get("longitude", 75.2951))
    ppm = float(zone_meta.get("px_per_meter", 90.0))

    # Subtle ROI perimeter outline
    if roi_polygon_px is not None and len(roi_polygon_px) >= 3:
        cv2.polylines(
            canvas,
            [roi_polygon_px.astype(np.int32)],
            isClosed=True,
            color=(230, 180, 40),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # 1. Draw vehicle corner-bracket boxes & license plate callout badges
    max_severity_score = float(pothole_summary.get("severity_index", 0.0))
    for veh in evaluated_vehicles:
        bbox = veh["bbox"]
        is_viol = bool(veh.get("is_violation", False))
        sev = veh.get("severity", "Low")
        if sev == "Critical":
            max_severity_score = max(max_severity_score, 92.0)
        elif sev == "High":
            max_severity_score = max(max_severity_score, 74.0)
        elif sev == "Medium":
            max_severity_score = max(max_severity_score, 52.0)

        color_bgr = (68, 68, 239) if is_viol else (129, 185, 16)
        _draw_corner_bracket_box(canvas, bbox, color_bgr=color_bgr, thickness=2, corner_len=18)

        x1, y1, x2, _ = [int(v) for v in bbox]
        plate_str = veh.get("formatted_plate") or veh.get("plate_number") or "UNREADABLE"
        inc_label = veh.get("incident_type", "Vehicle")
        ov_pct = float(veh.get("overlap_ratio", 0.0)) * 100.0
        badge_text = f"[{plate_str}] {inc_label} ({ov_pct:.0f}% ROI)"

        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.46, 1)
        by1 = max(38, y1 - th - 12)
        by2 = by1 + th + 8
        bx2 = min(w - 10, x1 + tw + 12)
        cv2.rectangle(canvas, (x1, by1), (bx2, by2), (24, 18, 15), -1)
        cv2.rectangle(canvas, (x1, by1), (bx2, by2), color_bgr, 1)
        cv2.putText(
            canvas,
            badge_text,
            (x1 + 6, by2 - 6),
            font,
            0.46,
            (250, 250, 255),
            1,
            cv2.LINE_AA,
        )

    # 2. Draw pothole tactical brackets & repair callouts
    for cav in pothole_cavities:
        cbbox = cav["bbox"]
        _draw_corner_bracket_box(canvas, cbbox, color_bgr=(0, 145, 255), thickness=2, corner_len=14)
        cx1, cy1, _, cy2 = cbbox
        p_label = (
            f"{cav['pothole_class']} | {cav['surface_area_sqm']:.2f}m2 | "
            f"SI:{cav['severity_index']:.0f}/100 | INR {cav['estimated_repair_cost_inr']:.0f}"
        )
        (tw, th), _ = cv2.getTextSize(p_label, font, 0.44, 1)
        ly1 = min(h - 65, cy2 + 6)
        ly2 = ly1 + th + 8
        cv2.rectangle(canvas, (cx1, ly1), (min(w - 10, cx1 + tw + 10), ly2), (20, 20, 30), -1)
        cv2.putText(
            canvas,
            p_label,
            (cx1 + 5, ly2 - 5),
            font,
            0.44,
            (56, 210, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Draw Severity Meter Bar (top-right HUD widget)
    meter_w, meter_h = 210, 44
    mx1, my1 = max(12, w - meter_w - 16), 46
    mx2, my2 = mx1 + meter_w, my1 + meter_h
    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (mx1, my1), (mx2, my2), (28, 22, 15), -1)
    cv2.addWeighted(hud_overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (200, 140, 30), 1)

    cv2.putText(
        canvas,
        f"SEVERITY INDEX: {max_severity_score:.1f} / 100",
        (mx1 + 10, my1 + 17),
        font,
        0.42,
        (240, 245, 250),
        1,
        cv2.LINE_AA,
    )
    bar_x1, bar_y1 = mx1 + 10, my1 + 25
    bar_w, bar_h = meter_w - 20, 10
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (65, 65, 75), -1)
    fill_w = int(round(np.clip(max_severity_score / 100.0, 0.0, 1.0) * bar_w))
    bar_col = (
        (68, 68, 239)
        if max_severity_score >= 70
        else (11, 158, 245)
        if max_severity_score >= 40
        else (129, 185, 16)
    )
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), bar_col, -1)

    # 4. Bottom Tactical Telemetry + GPS Watermark Bar
    footer_h = 34
    fy1 = h - footer_h
    f_overlay = canvas.copy()
    cv2.rectangle(f_overlay, (0, fy1), (w, h), (22, 16, 10), -1)
    cv2.addWeighted(f_overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.line(canvas, (0, fy1), (w, fy1), (199, 132, 2), 1)

    gps_text = (
        f"GPS: {lat:.4f}N, {lon:.4f}E | Scale: {ppm:.1f}px/m | CSMSS Kanchanwadi, Chhatrapati Sambhajinagar"
    )
    lat_text = (
        f"Pre:{latency_ms.get('preprocess_ms', 0.0):.1f}ms | "
        f"Det:{latency_ms.get('detection_ms', 0.0):.1f}ms | "
        f"ANPR:{latency_ms.get('anpr_ms', 0.0):.1f}ms | "
        f"Total:{latency_ms.get('total_ms', 0.0):.1f}ms"
    )
    cv2.putText(
        canvas,
        gps_text,
        (12, h - 11),
        font,
        0.42,
        (210, 225, 240),
        1,
        cv2.LINE_AA,
    )
    (ltw, _), _ = cv2.getTextSize(lat_text, font, 0.42, 1)
    cv2.putText(
        canvas,
        lat_text,
        (max(12, w - ltw - 12), h - 11),
        font,
        0.42,
        (129, 230, 110),
        1,
        cv2.LINE_AA,
    )

    _draw_top_banner(
        canvas,
        left_text=f"STAGE 4 [TACTICAL HUD] | {zone_id}: {zone_name}",
        right_text=f"{engine_mode} | {timestamp_str or _now_ist_display()}",
        accent_bgr=(68, 68, 239) if max_severity_score >= 70 else (215, 132, 2),
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def build_xray_visuals(
    frame_bgr: np.ndarray,
    zone_id: str,
    stage2_preprocessed_rgb: np.ndarray,
    roi_polygon_px: np.ndarray,
    combined_intersection_mask: np.ndarray,
    evaluated_vehicles: Sequence[Dict[str, Any]],
    pothole_summary: Dict[str, Any],
    pothole_cavities: Sequence[Dict[str, Any]],
    latency_ms: Dict[str, float],
    engine_mode: str,
    clahe_clip_limit: float = 2.5,
    timestamp_str: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate the complete 4-Stage X-Ray dictionary of RGB `np.ndarray` images:
    - `stage1_raw`
    - `stage2_preprocessed`
    - `stage3_geometry`
    - `stage4_hud`
    """
    ts = timestamp_str or _now_ist_display()
    s1 = render_stage1_raw(frame_bgr, zone_id=zone_id, timestamp_str=ts)
    s2 = render_stage2_preprocessed(
        stage2_preprocessed_rgb,
        zone_id=zone_id,
        clahe_clip_limit=clahe_clip_limit,
    )
    s3 = render_stage3_geometry(
        frame_bgr=frame_bgr,
        zone_id=zone_id,
        roi_polygon_px=roi_polygon_px,
        combined_intersection_mask=combined_intersection_mask,
        evaluated_vehicles=evaluated_vehicles,
        pothole_cavities=pothole_cavities,
    )
    s4 = render_stage4_hud(
        frame_bgr=frame_bgr,
        zone_id=zone_id,
        roi_polygon_px=roi_polygon_px,
        evaluated_vehicles=evaluated_vehicles,
        pothole_summary=pothole_summary,
        pothole_cavities=pothole_cavities,
        latency_ms=latency_ms,
        engine_mode=engine_mode,
        timestamp_str=ts,
    )
    return {
        "stage1_raw": s1,
        "stage2_preprocessed": s2,
        "stage3_geometry": s3,
        "stage4_hud": s4,
    }