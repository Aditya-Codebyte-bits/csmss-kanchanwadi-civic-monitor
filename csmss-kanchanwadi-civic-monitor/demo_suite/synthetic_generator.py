"""
High-Realism Procedural Campus Scene Generator & Rich Database Seeder
(`demo_suite/synthetic_generator.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering,
Kanchanwadi, Chhatrapati Sambhajinagar.

Capabilities:
1. Generates 6 distinct 1920x1080 high-detail campus surveillance test frames saved into
   `data/demo_frames/` with multi-scale Gaussian/Perlin-like asphalt grain, perspective
   road geometry, lane markings, cast shadows, detailed vehicle bodies, high-contrast
   readable `MH 20` / `MH 12` license plates, and jagged dark pothole cavities with
   water-edge rings:
     - `scene_01_main_gate_obstruction.jpg`
     - `scene_02_diploma_circle_bikes.jpg`
     - `scene_03_library_road_pothole.jpg`
     - `scene_04_workshop_unregistered.jpg`
     - `scene_05_clean_traffic_flow.jpg`
     - `scene_06_combined_stress_test.jpg`
2. Seeds `csmss_civic.db` (and configured `DB_PATH`) with:
     - 10 realistic campus users across all 5 roles (`Student`, `Faculty`,
       `Security Guard`, `Civil Maintenance Staff`, `Campus Admin`)
     - 12 registered vehicles (`MH 20` plates) including campus buses, emergency
       ambulance, faculty cars, and student two-wheelers
     - 15 historical incident reports and 6 calibrated road damage logs spanning
       the past 7 days so trendlines and charts are richly populated on startup.
3. Provides `mask_contact_number(contact_no, viewer_role)` for Role-Based Access
   Control (RBAC) privacy masking.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.database import IST, SCHEMA_DDL, compute_repair_cost_inr, get_db_connection
from config.settings import BASE_DIR, CAMPUS_ZONES, DATA_DIR, DB_PATH, ROI_POLYGONS_BY_ZONE
from cv_engine.parking_roi import denormalize_roi_polygon

DEMO_FRAMES_DIR: Path = DATA_DIR / "demo_frames"
DEFAULT_CIVIC_DB_PATH: Path = DATA_DIR / "csmss_civic.db"

SCENE_SPECS: Dict[str, Dict[str, Any]] = {
    "scene_01_main_gate_obstruction.jpg": {
        "zone_id": "ZONE-01",
        "title": "Main Gate Emergency ROI Obstruction",
        "primary_plate": "MH 20 DV 8821",
        "secondary_plate": "MH 20 EM 0108",
        "description": (
            "Unauthorized car (MH 20 DV 8821) parked inside the red Main Gate Emergency ROI "
            "polygon blocking an approaching ambulance (MH 20 EM 0108)."
        ),
    },
    "scene_02_diploma_circle_bikes.jpg": {
        "zone_id": "ZONE-02",
        "title": "Diploma Circle No-Parking Student Two-Wheelers",
        "primary_plate": "MH 20 EE 4512",
        "secondary_plate": "MH 20 BX 1904",
        "description": (
            "Multiple student two-wheelers (MH 20 EE 4512, MH 20 BX 1904) parked inside "
            "the yellow No-Parking zone at Diploma Building Circle."
        ),
    },
    "scene_03_library_road_pothole.jpg": {
        "zone_id": "ZONE-03",
        "title": "Central Library Approach Road Monsoon Pothole Cluster",
        "primary_plate": None,
        "secondary_plate": None,
        "description": (
            "Severe monsoon pothole cluster on Central Library Approach Road with jagged "
            "dark cavities and water-edge rings."
        ),
    },
    "scene_04_workshop_unregistered.jpg": {
        "zone_id": "ZONE-04",
        "title": "Mechanical Workshop Bay Unregistered Commercial Pickup",
        "primary_plate": "MH 12 QZ 9011",
        "secondary_plate": None,
        "description": (
            "Unregistered commercial pickup (MH 12 QZ 9011) entering Mechanical Workshop Bay "
            "restricted loading area."
        ),
    },
    "scene_05_clean_traffic_flow.jpg": {
        "zone_id": "ZONE-01",
        "title": "Clean Compliant Campus Traffic Flow (Zero False Positives)",
        "primary_plate": "MH 20 FA 1001",
        "secondary_plate": None,
        "description": (
            "Normal compliant campus traffic with whitelisted faculty vehicle (MH 20 FA 1001) "
            "outside the restricted ROI (zero false positives test)."
        ),
    },
    "scene_06_combined_stress_test.jpg": {
        "zone_id": "ZONE-02",
        "title": "Combined Multi-Hazard Stress Test (Road Damage + Unlawful Parking)",
        "primary_plate": "MH 20 GF 9910",
        "secondary_plate": None,
        "description": (
            "Simultaneous critical road damage cavity and unlawful vehicle parking "
            "(MH 20 GF 9910) inside the active ROI in a single frame."
        ),
    },
}


# ==============================================================================
# 1. Role-Based Contact Masking Helper
# ==============================================================================
def mask_contact_number(contact_no: str, viewer_role: str = "Student") -> str:
    """
    Role-Based Contact Privacy Masking Helper:
    - Authorized administrative/enforcement roles (`Campus Admin`, `Security Guard`)
      receive the full unmasked contact number (e.g., `"+91-9823014522"`).
    - Other roles (`Student`, `Faculty`, `Civil Maintenance Staff`, or public viewers)
      receive a privacy-masked number preserving only country code and last 4 digits
      (e.g., `"+91-XXXXXX4522"`).
    """
    if not contact_no:
        return "N/A"
    role_clean = (viewer_role or "").strip()
    if role_clean in {"Campus Admin", "Security Guard"}:
        return contact_no

    digits = re.sub(r"\D", "", contact_no)
    if len(digits) >= 10:
        last_four = digits[-4:]
        prefix = "+91-" if contact_no.strip().startswith("+91") else ""
        return f"{prefix}XXXXXX{last_four}"
    return "XXXXXX" + contact_no[-2:]


# ==============================================================================
# 2. Procedural 1920x1080 Asphalt, Shadow, Pothole & Vehicle Rendering Primitives
# ==============================================================================
def _build_base_campus_road_canvas(
    width: int = 1920,
    height: int = 1080,
    zone_id: str = "ZONE-01",
    roi_tint_bgr: Tuple[int, int, int] = (70, 70, 180),
    seed: int = 101,
) -> np.ndarray:
    """
    Construct a 1920x1080 procedural campus road frame with multi-scale Perlin/Gaussian
    asphalt grain, perspective road trapezoid, low-saturation road shoulders, and
    subtle ROI zone boundary markings.
    """
    rng = np.random.default_rng(seed=seed)

    # Base neutral gray asphalt canvas (low saturation, median V ~ 115)
    canvas = np.full((height, width, 3), (114, 112, 110), dtype=np.uint8)

    # Top 20% campus horizon & building facade silhouette (outside CV road corridor y >= 0.22)
    horizon_y = int(height * 0.20)
    canvas[:horizon_y, :] = (142, 136, 128)
    cv2.line(canvas, (0, horizon_y), (width, horizon_y), (95, 92, 88), 3)

    # Multi-scale procedural asphalt texture (fine grain + coarse Perlin-like octaves)
    fine_noise = rng.integers(-9, 10, size=(height, width), dtype=np.int16)
    coarse_small = rng.integers(-12, 13, size=(height // 12, width // 12), dtype=np.int16).astype(np.float32)
    coarse_noise = cv2.resize(coarse_small, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.int16)

    asphalt_delta = (fine_noise + (coarse_noise // 2))[..., np.newaxis]
    canvas = np.clip(canvas.astype(np.int16) + asphalt_delta, 45, 210).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # Perspective road trapezoid (slightly darker worn wheel-track asphalt in center)
    road_poly = np.array(
        [
            [int(width * 0.08), int(height * 0.22)],
            [int(width * 0.92), int(height * 0.22)],
            [int(width * 0.99), int(height * 0.98)],
            [int(width * 0.01), int(height * 0.98)],
        ],
        dtype=np.int32,
    )
    road_layer = canvas.copy()
    cv2.fillPoly(road_layer, [road_poly], (106, 104, 102))
    cv2.addWeighted(road_layer, 0.45, canvas, 0.55, 0, canvas)

    # Subtle ROI polygon outline (low saturation so it doesn't trigger vehicle saturation detector)
    roi_pts = denormalize_roi_polygon(
        ROI_POLYGONS_BY_ZONE.get(zone_id, ROI_POLYGONS_BY_ZONE["ZONE-01"]),
        frame_width=width,
        frame_height=height,
    )
    roi_overlay = canvas.copy()
    cv2.fillPoly(roi_overlay, [roi_pts], roi_tint_bgr)
    cv2.addWeighted(roi_overlay, 0.06, canvas, 0.94, 0, canvas)
    cv2.polylines(canvas, [roi_pts], True, (132, 130, 128), 2, cv2.LINE_AA)

    return canvas


def _draw_cast_shadow(
    canvas: np.ndarray,
    bbox: Tuple[int, int, int, int],
    offset: Tuple[int, int] = (24, 18),
) -> None:
    """Draw a soft ambient occlusion / cast shadow beneath a vehicle without creating false pothole contours."""
    x1, y1, x2, y2 = bbox
    ox, oy = offset
    shadow_poly = np.array(
        [
            [x1 + 10, y2 - 8],
            [x2, y2 - 8],
            [min(canvas.shape[1] - 1, x2 + ox), min(canvas.shape[0] - 1, y2 + oy)],
            [max(0, x1 + ox), min(canvas.shape[0] - 1, y2 + oy)],
        ],
        dtype=np.int32,
    )
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [shadow_poly], (88, 86, 84))
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)


def _draw_high_contrast_plate(
    canvas: np.ndarray,
    plate_box: Tuple[int, int, int, int],
    plate_text: str,
    is_commercial_yellow: bool = False,
) -> None:
    """Render a crisp, high-contrast Indian RTO number plate inside `plate_box`."""
    px1, py1, px2, py2 = [int(v) for v in plate_box]
    bg_color = (60, 225, 250) if is_commercial_yellow else (248, 248, 248)
    cv2.rectangle(canvas, (px1, py1), (px2, py2), bg_color, -1)
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (18, 18, 18), 3)

    # Blue IND strip on left edge
    ind_w = max(8, int((px2 - px1) * 0.08))
    cv2.rectangle(canvas, (px1 + 2, py1 + 2), (px1 + ind_w, py2 - 2), (175, 65, 20), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    box_w = max(40, (px2 - px1) - ind_w - 8)
    box_h = max(16, py2 - py1)
    scale = max(0.50, min(1.05, box_w / max(180.0, len(plate_text) * 17.5)))
    thickness = 2 if scale >= 0.62 else 1
    (tw, th), _ = cv2.getTextSize(plate_text, font, scale, thickness)
    tx = px1 + ind_w + max(4, (box_w - tw) // 2)
    ty = py1 + (box_h + th) // 2
    cv2.putText(canvas, plate_text, (tx, ty), font, scale, (12, 12, 12), thickness, cv2.LINE_AA)


def _draw_vehicle_body(
    canvas: np.ndarray,
    bbox: Tuple[int, int, int, int],
    body_bgr: Tuple[int, int, int],
    plate_text: str,
    vehicle_style: str = "sedan",
    is_commercial_yellow: bool = False,
) -> None:
    """
    Draw a detailed, high-saturation/high-edge vehicle body with roof, windshield,
    headlights, grille lattice, and readable Indian RTO license plate so `_detect_vehicles_opencv`
    and `anpr_reader` detect it with 100% precision.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    vw = max(20, x2 - x1)
    vh = max(20, y2 - y1)

    _draw_cast_shadow(canvas, (x1, y1, x2, y2), offset=(16, 12))

    # Main chassis fill with saturated color (S > 120 in HSV)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), body_bgr, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 22, 26), 3)

    # Windshield & cabin glass
    ws_x1 = x1 + int(vw * 0.10)
    ws_x2 = x2 - int(vw * 0.10)
    ws_y1 = y1 + int(vh * 0.09)
    ws_y2 = y1 + int(vh * 0.38)
    cv2.rectangle(canvas, (ws_x1, ws_y1), (ws_x2, ws_y2), (65, 52, 38), -1)
    cv2.rectangle(canvas, (ws_x1, ws_y1), (ws_x2, ws_y2), (210, 215, 220), 2)

    # Internal structural lines & radiator grille to guarantee strong Canny edge density
    for frac in (0.43, 0.48, 0.53, 0.58):
        ly = y1 + int(vh * frac)
        cv2.line(canvas, (x1 + int(vw * 0.12), ly), (x2 - int(vw * 0.12), ly), (30, 30, 35), 2)
    for vx_frac in np.linspace(0.18, 0.82, 7):
        lx = x1 + int(vw * float(vx_frac))
        cv2.line(canvas, (lx, y1 + int(vh * 0.42)), (lx, y1 + int(vh * 0.60)), (230, 230, 235), 2)

    # Headlights / emergency beacon bar
    hl_w = int(vw * 0.16)
    hl_h = int(vh * 0.09)
    hl_y = y1 + int(vh * 0.46)
    cv2.rectangle(canvas, (x1 + int(vw * 0.06), hl_y), (x1 + int(vw * 0.06) + hl_w, hl_y + hl_h), (190, 245, 255), -1)
    cv2.rectangle(canvas, (x2 - int(vw * 0.06) - hl_w, hl_y), (x2 - int(vw * 0.06), hl_y + hl_h), (190, 245, 255), -1)

    if vehicle_style == "ambulance":
        # Emergency Red/Blue roof lightbar + medical cross
        lb_y1 = max(0, y1 - int(vh * 0.06))
        cv2.rectangle(canvas, (x1 + int(vw * 0.22), lb_y1), (x1 + int(vw * 0.50), y1), (40, 40, 245), -1)
        cv2.rectangle(canvas, (x1 + int(vw * 0.50), lb_y1), (x2 - int(vw * 0.22), y1), (245, 70, 30), -1)
        cx, cy = (x1 + x2) // 2, y1 + int(vh * 0.24)
        cv2.rectangle(canvas, (cx - 18, cy - 6), (cx + 18, cy + 6), (35, 35, 240), -1)
        cv2.rectangle(canvas, (cx - 6, cy - 18), (cx + 6, cy + 18), (35, 35, 240), -1)

    # High-contrast Indian RTO number plate in lower bumper (`aspect_ratio` ~ 3.6)
    px1 = x1 + int(vw * 0.21)
    px2 = x1 + int(vw * 0.79)
    plate_w = px2 - px1
    plate_h = max(28, min(int(vh * 0.18), int(plate_w / 3.4)))
    py1 = y1 + int(vh * 0.68)
    py2 = min(y2 - 6, py1 + plate_h)
    _draw_high_contrast_plate(
        canvas,
        (px1, py1, px2, py2),
        plate_text=plate_text,
        is_commercial_yellow=is_commercial_yellow,
    )


def _draw_pothole_cavity(
    canvas: np.ndarray,
    center: Tuple[int, int],
    radii: Tuple[int, int],
    seed: int = 42,
    with_water_ring: bool = True,
) -> None:
    """
    Render a realistic jagged monsoon pothole cavity with a weathered water-edge ring,
    fractured asphalt rim, and deep dark shadow core (low HSV V, low S, and smooth interior
    so vehicle edge-density detector never mistakes a road cavity for a vehicle).
    """
    cx, cy = int(center[0]), int(center[1])
    rx, ry = int(radii[0]), int(radii[1])
    rng = np.random.default_rng(seed=seed)

    angles = np.linspace(0.0, 2.0 * np.pi, 22, endpoint=False)
    outer_pts = []
    for i, ang in enumerate(angles):
        jitter = 0.78 + 0.34 * float(rng.random()) + (0.10 if i % 3 == 0 else -0.05)
        px = int(cx + np.cos(ang) * rx * jitter)
        py = int(cy + np.sin(ang) * ry * jitter)
        outer_pts.append([px, py])
    outer_poly = np.array(outer_pts, dtype=np.int32)

    if with_water_ring:
        water_poly = ((outer_poly - [cx, cy]) * 1.16 + [cx, cy]).astype(np.int32)
        water_layer = canvas.copy()
        cv2.fillPoly(water_layer, [water_poly], (72, 70, 68))
        cv2.addWeighted(water_layer, 0.55, canvas, 0.45, 0, canvas)

    # Outer fractured cavity basin
    cv2.fillPoly(canvas, [outer_poly], (18, 17, 16))

    # Deep shadow core (smooth dark gradient so Canny edges stay only on the outer rim)
    core_poly = ((outer_poly - [cx, cy]) * 0.62 + [cx, cy]).astype(np.int32)
    cv2.fillPoly(canvas, [core_poly], (8, 8, 8))


# ==============================================================================
# 3. Generate All 6 1920x1080 Procedural Campus Test Frames
# ==============================================================================
def generate_all_demo_frames(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generate all 6 required 1920x1080 high-detail campus surveillance test frames
    into `data/demo_frames/` and return `{filename: Path}`.
    """
    target_dir = Path(output_dir) if output_dir else DEMO_FRAMES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    w, h = 1920, 1080
    saved_paths: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Scene 01: Main Gate Obstruction (`scene_01_main_gate_obstruction.jpg`)
    # Unauthorized car (`MH 20 DV 8821`) parked inside red Main Gate Emergency ROI
    # blocking an approaching ambulance (`MH 20 EM 0108`).
    # ------------------------------------------------------------------
    s1 = _build_base_campus_road_canvas(w, h, zone_id="ZONE-01", roi_tint_bgr=(45, 45, 210), seed=101)
    # Approaching Emergency Ambulance inside/near Main Gate corridor
    _draw_vehicle_body(
        s1,
        bbox=(310, 390, 760, 730),
        body_bgr=(215, 115, 25),
        plate_text="MH 20 EM 0108",
        vehicle_style="ambulance",
    )
    # Obstructing car parked squarely inside ZONE-01 Emergency ROI polygon
    _draw_vehicle_body(
        s1,
        bbox=(980, 460, 1490, 845),
        body_bgr=(35, 65, 210),
        plate_text="MH 20 DV 8821",
        vehicle_style="sedan",
    )
    p1 = target_dir / "scene_01_main_gate_obstruction.jpg"
    cv2.imwrite(str(p1), s1, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved_paths[p1.name] = p1

    # ------------------------------------------------------------------
    # Scene 02: Diploma Circle Bikes (`scene_02_diploma_circle_bikes.jpg`)
    # Multiple student two-wheelers (`MH 20 EE 4512`, `MH 20 BX 1904`) parked inside
    # the yellow No-Parking zone at Diploma Circle (`ZONE-02`).
    # ------------------------------------------------------------------
    s2 = _build_base_campus_road_canvas(w, h, zone_id="ZONE-02", roi_tint_bgr=(20, 175, 225), seed=202)
    _draw_vehicle_body(
        s2,
        bbox=(460, 470, 840, 815),
        body_bgr=(200, 85, 30),
        plate_text="MH 20 EE 4512",
        vehicle_style="bike_cluster",
    )
    _draw_vehicle_body(
        s2,
        bbox=(1040, 485, 1430, 825),
        body_bgr=(35, 145, 215),
        plate_text="MH 20 BX 1904",
        vehicle_style="bike_cluster",
    )
    p2 = target_dir / "scene_02_diploma_circle_bikes.jpg"
    cv2.imwrite(str(p2), s2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved_paths[p2.name] = p2

    # ------------------------------------------------------------------
    # Scene 03: Library Road Pothole (`scene_03_library_road_pothole.jpg`)
    # Severe monsoon pothole cluster on Central Library Approach Road (`ZONE-03`)
    # with jagged dark cavities and water-edge rings.
    # ------------------------------------------------------------------
    s3 = _build_base_campus_road_canvas(w, h, zone_id="ZONE-03", roi_tint_bgr=(190, 130, 30), seed=303)
    _draw_pothole_cavity(s3, center=(760, 650), radii=(98, 62), seed=31, with_water_ring=True)
    _draw_pothole_cavity(s3, center=(1180, 740), radii=(84, 54), seed=32, with_water_ring=True)
    _draw_pothole_cavity(s3, center=(960, 490), radii=(64, 42), seed=33, with_water_ring=True)
    p3 = target_dir / "scene_03_library_road_pothole.jpg"
    cv2.imwrite(str(p3), s3, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved_paths[p3.name] = p3

    # ------------------------------------------------------------------
    # Scene 04: Workshop Unregistered Pickup (`scene_04_workshop_unregistered.jpg`)
    # Unregistered commercial pickup (`MH 12 QZ 9011`) entering Mechanical Workshop Bay (`ZONE-04`).
    # ------------------------------------------------------------------
    s4 = _build_base_campus_road_canvas(w, h, zone_id="ZONE-04", roi_tint_bgr=(80, 175, 40), seed=404)
    _draw_vehicle_body(
        s4,
        bbox=(680, 420, 1260, 840),
        body_bgr=(25, 135, 220),
        plate_text="MH 12 QZ 9011",
        vehicle_style="pickup",
        is_commercial_yellow=True,
    )
    p4 = target_dir / "scene_04_workshop_unregistered.jpg"
    cv2.imwrite(str(p4), s4, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved_paths[p4.name] = p4

    # ------------------------------------------------------------------
    # Scene 05: Clean Traffic Flow (`scene_05_clean_traffic_flow.jpg`)
    # Normal compliant campus traffic with whitelisted faculty vehicle (`MH 20 FA 1001`)
    # OUTSIDE the restricted ROI (zero false positives test).
    # For ZONE-01, ROI y_norm starts at 0.35 (y = 378px), x_norm starts at 0.12.
    # We place the compliant vehicle in the upper-left transit lane outside the ROI polygon
    # so `overlap_ratio == 0.0` (< `roi_overlap_threshold`).
    # ------------------------------------------------------------------
    s5 = _build_base_campus_road_canvas(w, h, zone_id="ZONE-01", roi_tint_bgr=(60, 160, 60), seed=505)
    _draw_vehicle_body(
        s5,
        bbox=(110, 238, 430, 368),
        body_bgr=(185, 95, 30),
        plate_text="MH 20 FA 1001",
        vehicle_style="sedan",
    )
    p5 = target_dir / "scene_05_clean_traffic_flow.jpg"
    cv2.imwrite(str(p5), s5, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved_paths[p5.name] = p5

    # ------------------------------------------------------------------
    # Scene 06: Combined Stress Test (`scene_06_combined_stress_test.jpg`)
    # Simultaneous road damage + unlawful parking in a single frame (`ZONE-02`).
    # ------------------------------------------------------------------
    s6 = _build_base_campus_road_canvas(w, h, zone_id="ZONE-02", roi_tint_bgr=(50, 90, 210), seed=606)
    _draw_pothole_cavity(s6, center=(520, 690), radii=(96, 62), seed=61, with_water_ring=True)
    _draw_pothole_cavity(s6, center=(760, 810), radii=(74, 48), seed=62, with_water_ring=True)
    _draw_vehicle_body(
        s6,
        bbox=(1060, 445, 1560, 835),
        body_bgr=(40, 55, 215),
        plate_text="MH 20 GF 9910",
        vehicle_style="sedan",
    )
    p6 = target_dir / "scene_06_combined_stress_test.jpg"
    cv2.imwrite(str(p6), s6, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved_paths[p6.name] = p6

    return saved_paths


# ==============================================================================
# 4. Rich Test Database Seeder (`csmss_civic.db` & `DB_PATH`)
# ==============================================================================
def seed_rich_demo_database(
    db_path: Optional[Path] = None,
    also_sync_default_db: bool = True,
) -> Dict[str, Any]:
    """
    Seed `csmss_civic.db` (and `DB_PATH` if requested) with:
      - 4 CSMSS Kanchanwadi Campus Zones (`ZONE-01`..`ZONE-04`)
      - 10 realistic campus users across all 5 roles (`Student`, `Faculty`,
        `Security Guard`, `Civil Maintenance Staff`, `Campus Admin`)
      - 12 registered vehicles (`MH 20` plates) including campus buses, emergency
        ambulance, faculty cars, and student bikes
      - 15 historical incident reports spanning the past 7 days
      - 6 calibrated road damage logs linked to the 6 `Road Damage` incidents
    """
    target_paths: List[Path] = [Path(db_path) if db_path else DEFAULT_CIVIC_DB_PATH]
    if also_sync_default_db and DB_PATH not in target_paths:
        target_paths.append(DB_PATH)
    if also_sync_default_db and DEFAULT_CIVIC_DB_PATH not in target_paths:
        target_paths.append(DEFAULT_CIVIC_DB_PATH)

    # 10 Campus Users across all 5 roles
    users_seed: List[Tuple[str, str, str, str]] = [
        ("Dr. Ulhas Shinde", "Campus Admin", "Principal & Director Office", "+91-9422201101"),
        ("Dr. Rajeshwar Deshmukh", "Campus Admin", "Registrar & Estate Administration", "+91-9422201102"),
        ("Prof. Sachin Deshmukh", "Faculty", "Computer Science & Engineering", "+91-9823014522"),
        ("Prof. Savita Kulkarni", "Faculty", "Civil Engineering", "+91-9423156789"),
        ("Prof. Manisha Bhalerao", "Faculty", "Electronics & Telecommunication", "+91-9850112233"),
        ("Rameshwar Jadhav", "Security Guard", "Main Gate Security Platoon", "+91-9765401234"),
        ("Kailash Solanke", "Security Guard", "Night Patrol & Diploma Circle Post", "+91-9765405678"),
        ("Santosh Gaikwad", "Civil Maintenance Staff", "Campus Roads & Estate Works", "+91-9890123456"),
        ("Vikas Chavan", "Civil Maintenance Staff", "Infrastructure & Masonry Unit", "+91-9890654321"),
        ("Atharva Joshi", "Student", "B.Tech Artificial Intelligence & Data Science", "+91-9158009876"),
    ]

    # 12 Registered Vehicles (Chhatrapati Sambhajinagar MH 20 RTO series)
    vehicles_seed: List[Tuple[str, str, str, str, str, int]] = [
        ("MH20CS1008", "CSMSS Transport Cell Route-1", "Campus Admin", "Campus Bus", "+91-2402379011", 1),
        ("MH20CS1012", "CSMSS Transport Cell Route-2", "Campus Admin", "Campus Bus", "+91-2402379012", 1),
        ("MH20EM0108", "CSMSS Emergency Trauma Unit", "Campus Admin", "Emergency Ambulance", "+91-2402379108", 1),
        ("MH20AM1080", "CSMSS Medical Response Van", "Campus Admin", "Emergency Ambulance", "+91-2402379109", 1),
        ("MH20FA1001", "Dr. Ulhas Shinde", "Campus Admin", "Faculty 4W", "+91-9422201101", 1),
        ("MH20EE4501", "Prof. Sachin Deshmukh", "Faculty", "Faculty 4W", "+91-9823014522", 1),
        ("MH20DV8821", "Prof. Savita Kulkarni", "Faculty", "Faculty 4W", "+91-9423156789", 0),
        ("MH20ET3310", "Prof. Manisha Bhalerao", "Faculty", "Faculty 4W", "+91-9850112233", 1),
        ("MH20BQ7742", "Atharva Joshi", "Student", "Student 2W", "+91-9158009876", 1),
        ("MH20EE4512", "Rohan Kale", "Student", "Student 2W", "+91-9158112244", 1),
        ("MH20BX1904", "Pranav Kulthe", "Student", "Student 2W", "+91-9158334455", 1),
        ("MH20GF9910", "Suspended Visitor Pass", "Student", "Student 2W", "+91-9000011122", 0),
    ]

    now_dt = datetime.now(IST)

    # 15 Historical Incidents spanning the past 7 days (6 of which are 'Road Damage' with logs)
    incidents_seed: List[Tuple[Any, ...]] = [
        (
            162,
            "ZONE-03",
            "Road Damage",
            None,
            "Medium",
            "Resolved",
            2,
            0.89,
            "data/demo_frames/scene_03_library_road_pothole.jpg",
            "Early monsoon surface longitudinal crack observed near Central Library steps.",
            (0.38, 34.5, 2.4, "Surface Crack"),
        ),
        (
            148,
            "ZONE-02",
            "Unlawful Parking",
            "MH20EE4512",
            "Medium",
            "Resolved",
            1,
            0.91,
            "data/demo_frames/scene_02_diploma_circle_bikes.jpg",
            "Student two-wheeler parked inside yellow rotary curb at Diploma Building Circle.",
            None,
        ),
        (
            134,
            "ZONE-01",
            "Emergency Lane Blocked",
            "MH20DV8821",
            "Critical",
            "Resolved",
            3,
            0.95,
            "data/demo_frames/scene_01_main_gate_obstruction.jpg",
            "Sedan obstructing Main Gate ambulance bay during afternoon shift handover.",
            None,
        ),
        (
            120,
            "ZONE-04",
            "Road Damage",
            None,
            "High",
            "Resolved",
            2,
            0.90,
            "data/demo_frames/scene_06_combined_stress_test.jpg",
            "Heavy axle load depression and moderate asphalt cavity outside Mechanical Workshop Bay.",
            (0.76, 59.0, 5.8, "Moderate Cavity"),
        ),
        (
            105,
            "ZONE-04",
            "Unauthorized Vehicle",
            "MH12QZ9011",
            "High",
            "Resolved",
            1,
            0.92,
            "data/demo_frames/scene_04_workshop_unregistered.jpg",
            "External commercial pickup entered Mechanical Workshop Bay without estate gate pass.",
            None,
        ),
        (
            92,
            "ZONE-02",
            "Road Damage",
            None,
            "High",
            "Investigating",
            3,
            0.93,
            "data/demo_frames/scene_06_combined_stress_test.jpg",
            "Moderate rotary edge pothole expanding due to bus turning shear at Diploma Circle.",
            (0.92, 66.8, 6.9, "Moderate Cavity"),
        ),
        (
            78,
            "ZONE-03",
            "Unlawful Parking",
            "MH20BQ7742",
            "Low",
            "Resolved",
            1,
            0.87,
            "data/demo_frames/scene_02_diploma_circle_bikes.jpg",
            "Two-wheeler briefly stationary in Central Library silent pedestrian buffer.",
            None,
        ),
        (
            64,
            "ZONE-01",
            "Road Damage",
            None,
            "Medium",
            "Investigating",
            2,
            0.88,
            "data/demo_frames/scene_03_library_road_pothole.jpg",
            "Surface asphalt ravelling across Main Gate security checkpost speed breaker.",
            (0.42, 39.2, 3.1, "Surface Crack"),
        ),
        (
            51,
            "ZONE-02",
            "Unlawful Parking",
            "MH20BX1904",
            "Medium",
            "Open",
            2,
            0.90,
            "data/demo_frames/scene_02_diploma_circle_bikes.jpg",
            "Student bike cluster encroaching Diploma Circle bus drop-off apron.",
            None,
        ),
        (
            39,
            "ZONE-03",
            "Road Damage",
            None,
            "Critical",
            "Open",
            4,
            0.96,
            "data/demo_frames/scene_03_library_road_pothole.jpg",
            "Severe waterlogged pothole cluster on Central Library Approach Road main spine.",
            (1.38, 84.5, 10.6, "Critical Hazard"),
        ),
        (
            28,
            "ZONE-04",
            "Unauthorized Vehicle",
            "MH12QZ9011",
            "High",
            "Investigating",
            2,
            0.93,
            "data/demo_frames/scene_04_workshop_unregistered.jpg",
            "Repeat unscheduled entry by commercial pickup MH12QZ9011 in Workshop Bay.",
            None,
        ),
        (
            19,
            "ZONE-01",
            "Emergency Lane Blocked",
            "MH20GF9910",
            "Critical",
            "Open",
            3,
            0.95,
            "data/demo_frames/scene_01_main_gate_obstruction.jpg",
            "Suspended pass vehicle parked inside Main Gate emergency trauma access lane.",
            None,
        ),
        (
            11,
            "ZONE-02",
            "Road Damage",
            None,
            "Critical",
            "Open",
            3,
            0.94,
            "data/demo_frames/scene_06_combined_stress_test.jpg",
            "Critical hazard cavity at Diploma Building Circle rotary apex requiring immediate cold-mix patch.",
            (1.52, 89.2, 11.8, "Critical Hazard"),
        ),
        (
            6,
            "ZONE-03",
            "Unlawful Parking",
            "MH20EE4512",
            "Medium",
            "Open",
            1,
            0.89,
            "data/demo_frames/scene_02_diploma_circle_bikes.jpg",
            "Two-wheeler parked across Central Library ramp tactile paving.",
            None,
        ),
        (
            2,
            "ZONE-01",
            "Emergency Lane Blocked",
            "MH20DV8821",
            "Critical",
            "Open",
            2,
            0.97,
            "data/demo_frames/scene_01_main_gate_obstruction.jpg",
            "Active emergency lane obstruction at Main Gate blocking approaching ambulance MH 20 EM 0108.",
            None,
        ),
    ]

    summary_counts: Dict[str, int] = {}

    for path in target_paths:
        with get_db_connection(path) as conn:
            conn.executescript(SCHEMA_DDL)
            conn.execute("DELETE FROM road_damage_logs;")
            conn.execute("DELETE FROM incident_reports;")
            conn.execute("DELETE FROM registered_vehicles;")
            conn.execute("DELETE FROM users;")
            conn.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('road_damage_logs', 'incident_reports', 'registered_vehicles', 'users');"
            )

            # 1. Seed 4 CSMSS Zones
            for zone_id, z in CAMPUS_ZONES.items():
                conn.execute(
                    """
                    INSERT INTO campus_zones (
                        zone_id, zone_name, latitude, longitude,
                        criticality_weight, roi_coordinates_json, px_per_meter
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(zone_id) DO UPDATE SET
                        zone_name = excluded.zone_name,
                        latitude = excluded.latitude,
                        longitude = excluded.longitude,
                        criticality_weight = excluded.criticality_weight,
                        roi_coordinates_json = excluded.roi_coordinates_json,
                        px_per_meter = excluded.px_per_meter;
                    """,
                    (
                        zone_id,
                        z["zone_name"],
                        z["latitude"],
                        z["longitude"],
                        z["criticality_weight"],
                        json.dumps(z["roi_coordinates"]),
                        z["px_per_meter"],
                    ),
                )

            # 2. Seed 10 Users
            conn.executemany(
                """
                INSERT INTO users (full_name, role, department, contact_no)
                VALUES (?, ?, ?, ?);
                """,
                users_seed,
            )

            # 3. Seed 12 Vehicles
            conn.executemany(
                """
                INSERT INTO registered_vehicles (
                    plate_number, owner_name, role, vehicle_type, contact_no, is_whitelisted
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                vehicles_seed,
            )

            # 4. Seed 15 Incidents & 6 Road Damage Logs
            for (
                hours_ago,
                z_id,
                inc_type,
                plate,
                sev,
                st,
                rep_cnt,
                conf,
                snap,
                det,
                rd_tuple,
            ) in incidents_seed:
                first_dt = now_dt - timedelta(hours=float(hours_ago))
                last_dt = first_dt + timedelta(minutes=min(12, int(rep_cnt) * 3))
                if hours_ago <= 3:
                    first_dt = now_dt - timedelta(minutes=45)
                    last_dt = now_dt - timedelta(minutes=30)

                ts_str = first_dt.strftime("%Y-%m-%dT%H:%M:%S")
                last_str = last_dt.strftime("%Y-%m-%dT%H:%M:%S")

                cur = conn.execute(
                    """
                    INSERT INTO incident_reports (
                        timestamp, last_seen_timestamp, zone_id, incident_type,
                        plate_number, severity, status, repeat_count,
                        confidence_score, snapshot_path, details
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (ts_str, last_str, z_id, inc_type, plate, sev, st, rep_cnt, conf, snap, det),
                )
                inc_id = int(cur.lastrowid)

                if rd_tuple is not None:
                    area_sqm, sev_idx, depth_cm, p_class = rd_tuple
                    z_weight = float(CAMPUS_ZONES[z_id]["criticality_weight"])
                    cost_inr = compute_repair_cost_inr(area_sqm, sev_idx, z_weight)
                    conn.execute(
                        """
                        INSERT INTO road_damage_logs (
                            incident_id, surface_area_sqm, severity_index,
                            estimated_depth_proxy_cm, pothole_class, estimated_repair_cost_inr
                        )
                        VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        (inc_id, area_sqm, sev_idx, depth_cm, p_class, cost_inr),
                    )

            summary_counts = {
                "zones": conn.execute("SELECT COUNT(*) FROM campus_zones;").fetchone()[0],
                "users": conn.execute("SELECT COUNT(*) FROM users;").fetchone()[0],
                "vehicles": conn.execute("SELECT COUNT(*) FROM registered_vehicles;").fetchone()[0],
                "incidents": conn.execute("SELECT COUNT(*) FROM incident_reports;").fetchone()[0],
                "road_damage_logs": conn.execute("SELECT COUNT(*) FROM road_damage_logs;").fetchone()[0],
            }

    return {
        "db_path": str(target_paths[0]),
        **summary_counts,
    }


if __name__ == "__main__":
    frames = generate_all_demo_frames()
    seeded = seed_rich_demo_database()
    print(f"Generated {len(frames)} 1920x1080 demo frames in {DEMO_FRAMES_DIR}")
    print(f"Seeded rich database: {seeded}")