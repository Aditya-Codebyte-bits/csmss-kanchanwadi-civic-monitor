"""
Module 2: Backend Server & SQLite Persistence Layer (app.py)
Project: CSMSS Chh. Shahu College of Engineering (Kanchanwadi) Civic Monitoring System
Course: Community Involved Project (CIP - 150 Marks)

Description:
FastAPI backend implementing the 5-table SQLite relational schema:
  - `users`: Students, faculty, security guards, civil maintenance staff
  - `campus_zones`: Kanchanwadi campus GPS coordinates, criticality weights & ROI boxes
  - `registered_vehicles`: Whitelisted student/faculty/bus/ambulance vehicles
  - `incident_reports`: Master geo-tagged civic & security incident log
  - `road_damage_logs`: Pothole & surface fracture maintenance extension records

Exposes REST API endpoints (with interactive Swagger UI at `/docs`) for:
  - Automated image analysis (Unlawful Parking, Road Damage, Gate ANPR)
  - License plate whitelist lookup (`GET /api/v1/vehicles/verify/{plate_number}`)
  - Geo-tagged incident querying, heatmap metrics, and Admin resolution workflow
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import CSMSSInferenceEngine

# Configurable storage directory (supports zero-cost local execution)
DATA_DIR = Path(os.environ.get("CSMSS_DATA_DIR", "./data"))
UPLOAD_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
DB_PATH = DATA_DIR / "csmss_civic.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Initialize unified CV engine
ml_engine = CSMSSInferenceEngine(output_dir=str(ANNOTATED_DIR))

app = FastAPI(
    title="CSMSS Kanchanwadi Smart Civic Monitoring API",
    description=(
        "Zero-Cost AI Civic & Security Monitoring Backend for CSMSS Chh. Shahu College "
        "of Engineering, Kanchanwadi, Chhatrapati Sambhajinagar (CIP 150-Mark Project)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Database Initialization & Connection Helper
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = MEMORY;
PRAGMA synchronous = OFF;

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('student','faculty','security','admin','civil_cell')),
    department_or_id TEXT UNIQUE NOT NULL,
    contact_phone TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS campus_zones (
    zone_id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_code TEXT UNIQUE NOT NULL,
    zone_name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    is_no_parking_zone BOOLEAN DEFAULT 1,
    criticality_weight REAL DEFAULT 1.0,
    roi_polygon_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS registered_vehicles (
    vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT UNIQUE NOT NULL,
    owner_name TEXT NOT NULL,
    owner_role TEXT NOT NULL CHECK(owner_role IN ('student','faculty','staff','college_bus','ambulance')),
    department TEXT NOT NULL,
    vehicle_type TEXT NOT NULL CHECK(vehicle_type IN ('two_wheeler','four_wheeler','bus','emergency')),
    pass_sticker_id TEXT UNIQUE NOT NULL,
    contact_phone TEXT,
    is_active BOOLEAN DEFAULT 1
);

CREATE TABLE IF NOT EXISTS incident_reports (
    incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_type TEXT NOT NULL CHECK(incident_type IN ('unlawful_parking','road_damage','unauthorized_vehicle')),
    zone_id INTEGER NOT NULL,
    reported_by_user_id INTEGER NOT NULL,
    detected_plate TEXT,
    is_authorized_vehicle BOOLEAN,
    confidence_score REAL NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('Low','Medium','High','Critical')),
    status TEXT NOT NULL CHECK(status IN ('Open','In_Progress','Resolved','Dismissed')) DEFAULT 'Open',
    image_path TEXT NOT NULL,
    description TEXT NOT NULL,
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (zone_id) REFERENCES campus_zones(zone_id),
    FOREIGN KEY (reported_by_user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS road_damage_logs (
    damage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL UNIQUE,
    defect_category TEXT NOT NULL CHECK(defect_category IN ('Pothole','Alligator_Crack','Longitudinal_Crack','Waterlogging_Edge_Drop')),
    bbox_area_ratio REAL NOT NULL,
    estimated_diameter_cm REAL NOT NULL,
    recommended_action TEXT NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incident_reports(incident_id) ON DELETE CASCADE
);
"""


def _open_sqlite_conn() -> sqlite3.Connection:
    """Opens SQLite connection with portable journal settings for local/USB/WSL filesystems."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("PRAGMA synchronous = OFF;")
    return conn


def init_db() -> None:
    """Creates all 5 SQLite tables if they do not already exist."""
    conn = _open_sqlite_conn()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yields a SQLite connection with dictionary-like Row factory."""
    init_db()
    conn = _open_sqlite_conn()
    try:
        yield conn
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class IncidentStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(Open|In_Progress|Resolved|Dismissed)$")
    resolution_notes: str = Field(..., min_length=3)


class VehicleCreateRequest(BaseModel):
    plate_number: str
    owner_name: str
    owner_role: str = "student"
    department: str
    vehicle_type: str = "two_wheeler"
    pass_sticker_id: str
    contact_phone: Optional[str] = None


# -----------------------------------------------------------------------------
# Helper Functions for Business Logic
# -----------------------------------------------------------------------------
def lookup_vehicle(conn: sqlite3.Connection, plate_number: Optional[str]) -> Optional[Dict[str, Any]]:
    """Checks if a normalized license plate exists in `registered_vehicles`."""
    if not plate_number:
        return None
    norm = CSMSSInferenceEngine.normalize_indian_plate(plate_number) or plate_number.upper().strip()
    cur = conn.execute(
        "SELECT * FROM registered_vehicles WHERE plate_number = ? AND is_active = 1",
        (norm,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_zone_or_default(conn: sqlite3.Connection, zone_id: int) -> Dict[str, Any]:
    """Fetches a campus zone by ID or returns the first available CSMSS zone."""
    cur = conn.execute("SELECT * FROM campus_zones WHERE zone_id = ?", (zone_id,))
    row = cur.fetchone()
    if row:
        return dict(row)
    cur_any = conn.execute("SELECT * FROM campus_zones ORDER BY zone_id ASC LIMIT 1")
    fallback = cur_any.fetchone()
    if fallback:
        return dict(fallback)
    raise HTTPException(status_code=404, detail="No campus zones configured. Run seed_and_test.py first.")


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    """Health and ML engine diagnostic endpoint."""
    return {
        "status": "healthy",
        "campus": "CSMSS Chh. Shahu College of Engineering, Kanchanwadi",
        "database_path": str(DB_PATH),
        "ml_engine_mode": ml_engine.engine_mode,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/zones")
def list_campus_zones() -> List[Dict[str, Any]]:
    """Returns all pre-configured CSMSS Kanchanwadi campus zones and GPS coordinates."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM campus_zones ORDER BY zone_id ASC").fetchall()
        return [dict(r) for r in rows]


@app.get("/api/v1/vehicles")
def list_registered_vehicles(
    role: Optional[str] = Query(None, description="Filter by owner_role (student, faculty, etc.)"),
) -> List[Dict[str, Any]]:
    """Returns whitelisted CSMSS vehicles."""
    with get_db() as conn:
        if role:
            rows = conn.execute(
                "SELECT * FROM registered_vehicles WHERE owner_role = ? ORDER BY vehicle_id DESC",
                (role,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM registered_vehicles ORDER BY vehicle_id DESC").fetchall()
        return [dict(r) for r in rows]


@app.post("/api/v1/vehicles")
def register_new_vehicle(payload: VehicleCreateRequest) -> Dict[str, Any]:
    """Adds a new student or staff vehicle to the authorized CSMSS registry."""
    norm_plate = CSMSSInferenceEngine.normalize_indian_plate(payload.plate_number) or payload.plate_number.upper()
    with get_db() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO registered_vehicles
                (plate_number, owner_name, owner_role, department, vehicle_type, pass_sticker_id, contact_phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    norm_plate,
                    payload.owner_name,
                    payload.owner_role,
                    payload.department,
                    payload.vehicle_type,
                    payload.pass_sticker_id,
                    payload.contact_phone,
                ),
            )
            conn.commit()
            return {"status": "registered", "vehicle_id": cur.lastrowid, "plate_number": norm_plate}
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=400, detail=f"Vehicle or pass sticker already exists: {exc}")


@app.get("/api/v1/vehicles/verify/{plate_number}")
def verify_vehicle_plate(plate_number: str) -> Dict[str, Any]:
    """Cross-references a license plate against the authorized CSMSS vehicle registry."""
    with get_db() as conn:
        record = lookup_vehicle(conn, plate_number)
        norm = CSMSSInferenceEngine.normalize_indian_plate(plate_number) or plate_number.upper()
        if record:
            return {
                "plate_number": norm,
                "is_authorized": True,
                "classification": "AUTHORIZED_CSMSS_MEMBER",
                "vehicle_details": record,
            }
        return {
            "plate_number": norm,
            "is_authorized": False,
            "classification": "UNREGISTERED_OUTSIDER",
            "vehicle_details": None,
        }


@app.post("/api/v1/analyze/{incident_type}")
async def analyze_and_log_incident(
    incident_type: str,
    zone_id: int = Form(1),
    reported_by_user_id: int = Form(1),
    hint_plate: Optional[str] = Form(None),
    user_notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Core automated analysis endpoint:
      1. Saves uploaded campus image frame.
      2. Runs the requested ML pipeline (`unlawful_parking`, `road_damage`, or `unauthorized_vehicle`).
      3. Verifies extracted license plate against `registered_vehicles` (if applicable).
      4. Logs geo-tagged incident into `incident_reports` (and `road_damage_logs` for road defects).
    """
    if incident_type not in ("unlawful_parking", "road_damage", "unauthorized_vehicle"):
        raise HTTPException(status_code=400, detail=f"Unsupported incident_type: {incident_type}")

    safe_filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    saved_upload_path = UPLOAD_DIR / safe_filename
    with open(saved_upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with get_db() as conn:
        zone = get_zone_or_default(conn, zone_id)
        roi_coords = tuple(json.loads(zone["roi_polygon_json"]))
        crit_weight = float(zone["criticality_weight"])

        if incident_type == "unlawful_parking":
            res = ml_engine.detect_unlawful_parking(
                str(saved_upload_path),
                roi_normalized=roi_coords,
                zone_criticality=crit_weight,
                hint_plate=hint_plate,
            )
        elif incident_type == "road_damage":
            res = ml_engine.detect_road_damage(
                str(saved_upload_path),
                zone_criticality=crit_weight,
            )
        else:
            res = ml_engine.extract_license_plate(
                str(saved_upload_path),
                hint_plate=hint_plate,
            )

        # Check plate authorization if a plate was detected
        vehicle_record = lookup_vehicle(conn, res.detected_plate)
        is_authorized: Optional[bool] = None
        if res.detected_plate:
            is_authorized = bool(vehicle_record is not None)
            if incident_type == "unauthorized_vehicle":
                if not is_authorized:
                    res.severity = "High" if crit_weight < 1.5 else "Critical"
                    res.summary = (
                        f"ALERT: Unregistered outsider vehicle ({res.detected_plate}) "
                        f"detected at {zone['zone_name']}."
                    )
                else:
                    res.severity = "Low"
                    res.summary = (
                        f"Authorized {vehicle_record['owner_role']} vehicle ({res.detected_plate}) "
                        f"owned by {vehicle_record['owner_name']} ({vehicle_record['department']})."
                    )

        full_description = res.summary
        if user_notes:
            full_description = f"{res.summary} | Reporter Note: {user_notes}"

        cur = conn.execute(
            """
            INSERT INTO incident_reports
            (incident_type, zone_id, reported_by_user_id, detected_plate, is_authorized_vehicle,
             confidence_score, severity, status, image_path, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Open', ?, ?)
            """,
            (
                incident_type,
                zone["zone_id"],
                reported_by_user_id,
                res.detected_plate,
                is_authorized,
                res.confidence,
                res.severity,
                res.annotated_image_path,
                full_description,
            ),
        )
        incident_id = cur.lastrowid

        # If road_damage, also populate specialized `road_damage_logs` table
        if incident_type == "road_damage":
            conn.execute(
                """
                INSERT INTO road_damage_logs
                (incident_id, defect_category, bbox_area_ratio, estimated_diameter_cm, recommended_action)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    res.defect_category or "Pothole",
                    res.bbox_area_ratio,
                    res.estimated_diameter_cm,
                    res.recommended_action or "Inspect road surface.",
                ),
            )

        conn.commit()

        return {
            "incident_id": incident_id,
            "zone": zone,
            "inference": res.to_dict(),
            "is_authorized_vehicle": is_authorized,
            "registered_vehicle_info": vehicle_record,
        }


@app.get("/api/v1/incidents")
def list_incidents(
    incident_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    zone_id: Optional[int] = Query(None),
) -> List[Dict[str, Any]]:
    """
    Queries geo-tagged campus incidents joined with zone coordinates, reporter details,
    registered vehicle lookup, and optional road damage metrics.
    """
    query = """
        SELECT
            ir.*,
            cz.zone_code,
            cz.zone_name,
            cz.latitude,
            cz.longitude,
            u.full_name AS reporter_name,
            u.role AS reporter_role,
            rv.owner_name AS vehicle_owner_name,
            rv.department AS vehicle_owner_dept,
            rv.contact_phone AS vehicle_owner_phone,
            rd.defect_category,
            rd.bbox_area_ratio,
            rd.estimated_diameter_cm,
            rd.recommended_action
        FROM incident_reports ir
        JOIN campus_zones cz ON ir.zone_id = cz.zone_id
        JOIN users u ON ir.reported_by_user_id = u.user_id
        LEFT JOIN registered_vehicles rv ON ir.detected_plate = rv.plate_number
        LEFT JOIN road_damage_logs rd ON ir.incident_id = rd.incident_id
        WHERE 1=1
    """
    params: List[Any] = []
    if incident_type:
        query += " AND ir.incident_type = ?"
        params.append(incident_type)
    if status:
        query += " AND ir.status = ?"
        params.append(status)
    if severity:
        query += " AND ir.severity = ?"
        params.append(severity)
    if zone_id:
        query += " AND ir.zone_id = ?"
        params.append(zone_id)

    query += " ORDER BY ir.created_at DESC, ir.incident_id DESC"

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


@app.patch("/api/v1/incidents/{incident_id}/resolve")
def update_incident_status(incident_id: int, payload: IncidentStatusUpdate) -> Dict[str, Any]:
    """Admin/Security workflow endpoint to update incident status and record resolution notes."""
    resolved_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") if payload.status == "Resolved" else None
    with get_db() as conn:
        cur = conn.execute(
            """
            UPDATE incident_reports
            SET status = ?, resolution_notes = ?, resolved_at = COALESCE(?, resolved_at)
            WHERE incident_id = ?
            """,
            (payload.status, payload.resolution_notes, resolved_ts, incident_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Incident #{incident_id} not found.")
        updated = conn.execute(
            "SELECT * FROM incident_reports WHERE incident_id = ?", (incident_id,)
        ).fetchone()
        return dict(updated)


@app.get("/api/v1/analytics/summary")
def get_campus_analytics() -> Dict[str, Any]:
    """Aggregates KPI metrics for the CSMSS Admin Command Center and CIP impact evaluation."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM incident_reports").fetchone()[0]
        open_cnt = conn.execute(
            "SELECT COUNT(*) FROM incident_reports WHERE status IN ('Open', 'In_Progress')"
        ).fetchone()[0]
        resolved_cnt = conn.execute(
            "SELECT COUNT(*) FROM incident_reports WHERE status = 'Resolved'"
        ).fetchone()[0]
        outsiders = conn.execute(
            "SELECT COUNT(*) FROM incident_reports WHERE is_authorized_vehicle = 0"
        ).fetchone()[0]

        by_type = conn.execute(
            "SELECT incident_type, COUNT(*) as count FROM incident_reports GROUP BY incident_type"
        ).fetchall()

        by_zone = conn.execute(
            """
            SELECT cz.zone_name, cz.latitude, cz.longitude, COUNT(ir.incident_id) as incident_count
            FROM campus_zones cz
            LEFT JOIN incident_reports ir ON cz.zone_id = ir.zone_id
            GROUP BY cz.zone_id
            ORDER BY incident_count DESC
            """
        ).fetchall()

        return {
            "total_incidents": total,
            "active_open_incidents": open_cnt,
            "resolved_incidents": resolved_cnt,
            "unauthorized_outsider_flags": outsiders,
            "breakdown_by_type": {row["incident_type"]: row["count"] for row in by_type},
            "zone_heatmap_summary": [dict(row) for row in by_zone],
        }


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
