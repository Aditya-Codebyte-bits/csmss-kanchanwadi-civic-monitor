"""
Shared Configuration & Constants for CSMSS Kanchanwadi Civic Monitor v2.0.
CSMSS Chh. Shahu College of Engineering, Kanchanwadi, Chhatrapati Sambhajinagar.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ==============================================================================
# 1. Directory & Storage Paths
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = Path(os.getenv("CSMSS_DATA_DIR", str(BASE_DIR / "data")))
SNAPSHOT_DIR: Path = DATA_DIR / "snapshots"
DB_PATH: Path = Path(os.getenv("CSMSS_DB_PATH", str(DATA_DIR / "csmss_civic_monitor.db")))

DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Application & Engine Settings
# ==============================================================================
APP_TITLE: str = "CSMSS Kanchanwadi Civic Monitor v2.0"
APP_VERSION: str = "2.0.0"
INSTITUTION_NAME: str = (
    "CSMSS Chh. Shahu College of Engineering, Kanchanwadi, Chhatrapati Sambhajinagar"
)
CAMPUS_CENTER_LAT: float = 19.8260
CAMPUS_CENTER_LON: float = 75.2950

# Engine Mode: "deterministic_cv" (Stage 1 CPU OpenCV) or "hybrid_dl" (Stage 2 YOLOv8 + EasyOCR)
ENGINE_MODE: str = os.getenv("CSMSS_ENGINE_MODE", "deterministic_cv")

# Temporal Deduplication Settings
DEDUP_WINDOW_MINUTES: int = int(os.getenv("CSMSS_DEDUP_WINDOW_MINUTES", "15"))

# Road Damage Heuristic Calibration Constants
BASE_REPAIR_RATE_INR_PER_SQM: float = 1850.0
MIN_REPAIR_MOBILIZATION_COST_INR: float = 450.0

# ==============================================================================
# 3. Campus Zones Configuration (4 Real CSMSS Kanchanwadi Zones)
# ==============================================================================
# Normalized ROI Polygon coordinates [(x_norm, y_norm), ...] in [0.0, 1.0] frame space
ROI_POLYGONS_BY_ZONE: Dict[str, List[Tuple[float, float]]] = {
    "ZONE-01": [(0.12, 0.35), (0.88, 0.35), (0.96, 0.92), (0.04, 0.92)],
    "ZONE-02": [(0.15, 0.30), (0.85, 0.30), (0.94, 0.90), (0.06, 0.90)],
    "ZONE-03": [(0.18, 0.28), (0.82, 0.28), (0.92, 0.94), (0.08, 0.94)],
    "ZONE-04": [(0.10, 0.32), (0.90, 0.32), (0.95, 0.92), (0.05, 0.92)],
}

# Perspective homography scale constants (calibrated pixels per physical meter at road plane)
PIXELS_PER_METER_BY_ZONE: Dict[str, float] = {
    "ZONE-01": 92.0,  # Main Gate camera mounted at 4.2m height, wide angle
    "ZONE-02": 84.5,  # Diploma Building Circle pole camera at 4.8m height
    "ZONE-03": 78.0,  # Central Library straight corridor camera at 5.0m height
    "ZONE-04": 96.0,  # Mechanical Workshop Bay lower gantry camera at 3.8m height
}

CAMPUS_ZONES: Dict[str, Dict[str, Any]] = {
    "ZONE-01": {
        "zone_id": "ZONE-01",
        "zone_name": "Main Gate & Security Plaza",
        "latitude": 19.8268,
        "longitude": 75.2951,
        "criticality_weight": 1.5,
        "roi_coordinates": ROI_POLYGONS_BY_ZONE["ZONE-01"],
        "px_per_meter": PIXELS_PER_METER_BY_ZONE["ZONE-01"],
        "description": "Primary Paithan Road link entry, emergency ambulance access & security checkpost.",
    },
    "ZONE-02": {
        "zone_id": "ZONE-02",
        "zone_name": "Diploma Building Circle",
        "latitude": 19.8262,
        "longitude": 75.2944,
        "criticality_weight": 1.3,
        "roi_coordinates": ROI_POLYGONS_BY_ZONE["ZONE-02"],
        "px_per_meter": PIXELS_PER_METER_BY_ZONE["ZONE-02"],
        "description": "High-density pedestrian rotary and bus drop-off junction near polytechnic wing.",
    },
    "ZONE-03": {
        "zone_id": "ZONE-03",
        "zone_name": "Central Library Approach Road",
        "latitude": 19.8256,
        "longitude": 75.2958,
        "criticality_weight": 1.2,
        "roi_coordinates": ROI_POLYGONS_BY_ZONE["ZONE-03"],
        "px_per_meter": PIXELS_PER_METER_BY_ZONE["ZONE-03"],
        "description": "Silent academic spine connecting Central Library and main administrative block.",
    },
    "ZONE-04": {
        "zone_id": "ZONE-04",
        "zone_name": "Mechanical Workshop Bay",
        "latitude": 19.8251,
        "longitude": 75.2949,
        "criticality_weight": 1.1,
        "roi_coordinates": ROI_POLYGONS_BY_ZONE["ZONE-04"],
        "px_per_meter": PIXELS_PER_METER_BY_ZONE["ZONE-04"],
        "description": "Heavy lab equipment loading zone and service vehicle turnaround bay.",
    },
}

# ==============================================================================
# 4. Domain Vocabularies & Enumerations
# ==============================================================================
VALID_USER_ROLES: Tuple[str, ...] = (
    "Student",
    "Faculty",
    "Security Guard",
    "Civil Maintenance Staff",
    "Campus Admin",
)

VALID_VEHICLE_TYPES: Tuple[str, ...] = (
    "Student 2W",
    "Faculty 4W",
    "Campus Bus",
    "Emergency Ambulance",
)

VALID_INCIDENT_TYPES: Tuple[str, ...] = (
    "Unlawful Parking",
    "Road Damage",
    "Unauthorized Vehicle",
    "Emergency Lane Blocked",
)

VALID_SEVERITIES: Tuple[str, ...] = (
    "Low",
    "Medium",
    "High",
    "Critical",
)

VALID_STATUSES: Tuple[str, ...] = (
    "Open",
    "Investigating",
    "Resolved",
)

VALID_POTHOLE_CLASSES: Tuple[str, ...] = (
    "Surface Crack",
    "Moderate Cavity",
    "Critical Hazard",
)

# ==============================================================================
# 5. UI Color Tokens & Visual Styling Palette
# ==============================================================================
UI_COLOR_TOKENS: Dict[str, Any] = {
    "brand": {
        "primary": "#1E3A8A",       # CSMSS Deep Royal Blue
        "secondary": "#D97706",     # Maratha Heritage Saffron/Amber
        "accent": "#0284C7",        # Civic Telemetry Cyan
        "surface_dark": "#0F172A",  # Slate 900
        "surface_card": "#1E293B",  # Slate 800
        "text_primary": "#F8FAFC",
        "text_muted": "#94A3B8",
    },
    "severity": {
        "Low": "#10B981",           # Emerald Green
        "Medium": "#F59E0B",        # Amber Gold
        "High": "#F97316",          # Deep Orange
        "Critical": "#EF4444",      # Crimson Alert
    },
    "status": {
        "Open": "#EF4444",
        "Investigating": "#F59E0B",
        "Resolved": "#10B981",
    },
    "pothole_class": {
        "Surface Crack": "#38BDF8",
        "Moderate Cavity": "#F59E0B",
        "Critical Hazard": "#EF4444",
    },
    "zone_rgb": {
        "ZONE-01": [239, 68, 68, 200],    # High priority red tint
        "ZONE-02": [245, 158, 11, 200],   # Amber tint
        "ZONE-03": [14, 165, 233, 200],   # Sky blue tint
        "ZONE-04": [16, 185, 129, 200],   # Emerald tint
    },
}