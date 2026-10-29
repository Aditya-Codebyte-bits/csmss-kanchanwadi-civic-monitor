"""
SQLite Relational Database Layer (5-Table Schema) for CSMSS Kanchanwadi Civic Monitor v2.0.
Includes automatic schema initialization, CSMSS campus seed data, and query helpers.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from config.settings import (
    BASE_REPAIR_RATE_INR_PER_SQM,
    CAMPUS_ZONES,
    DB_PATH,
    MIN_REPAIR_MOBILIZATION_COST_INR,
)

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist_iso() -> str:
    """Return current Indian Standard Time timestamp in ISO-8601 format (YYYY-MM-DDTHH:MM:SS)."""
    return datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S")


def compute_repair_cost_inr(
    surface_area_sqm: float,
    severity_index: float,
    criticality_weight: float = 1.2,
) -> float:
    """
    Calculate estimated civil repair cost in INR based on calibrated area (sqm),
    0-100 severity index, and campus zone criticality weight.
    """
    severity_multiplier = 1.0 + (max(0.0, min(100.0, severity_index)) / 100.0) * 0.85
    raw_cost = (
        MIN_REPAIR_MOBILIZATION_COST_INR
        + surface_area_sqm * BASE_REPAIR_RATE_INR_PER_SQM * severity_multiplier * criticality_weight
    )
    return round(raw_cost, 2)


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context-managed SQLite connection with Row factory and foreign key enforcement."""
    target_path = Path(db_path) if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=15.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ==============================================================================
# 1. 5-Table Relational DDL Schema
# ==============================================================================
SCHEMA_DDL: str = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (
        role IN ('Student', 'Faculty', 'Security Guard', 'Civil Maintenance Staff', 'Campus Admin')
    ),
    department TEXT NOT NULL,
    contact_no TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campus_zones (
    zone_id TEXT PRIMARY KEY,
    zone_name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    criticality_weight REAL NOT NULL DEFAULT 1.0,
    roi_coordinates_json TEXT NOT NULL,
    px_per_meter REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS registered_vehicles (
    vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL UNIQUE,
    owner_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (
        role IN ('Student', 'Faculty', 'Security Guard', 'Civil Maintenance Staff', 'Campus Admin')
    ),
    vehicle_type TEXT NOT NULL CHECK (
        vehicle_type IN ('Student 2W', 'Faculty 4W', 'Campus Bus', 'Emergency Ambulance')
    ),
    contact_no TEXT NOT NULL,
    is_whitelisted INTEGER NOT NULL DEFAULT 1 CHECK (is_whitelisted IN (0, 1))
);

CREATE TABLE IF NOT EXISTS incident_reports (
    incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    last_seen_timestamp TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    incident_type TEXT NOT NULL CHECK (
        incident_type IN ('Unlawful Parking', 'Road Damage', 'Unauthorized Vehicle', 'Emergency Lane Blocked')
    ),
    plate_number TEXT,
    severity TEXT NOT NULL CHECK (
        severity IN ('Low', 'Medium', 'High', 'Critical')
    ),
    status TEXT NOT NULL DEFAULT 'Open' CHECK (
        status IN ('Open', 'Investigating', 'Resolved')
    ),
    repeat_count INTEGER NOT NULL DEFAULT 1,
    confidence_score REAL NOT NULL DEFAULT 0.85,
    snapshot_path TEXT,
    details TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (zone_id) REFERENCES campus_zones(zone_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS road_damage_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL UNIQUE,
    surface_area_sqm REAL NOT NULL,
    severity_index REAL NOT NULL CHECK (severity_index >= 0.0 AND severity_index <= 100.0),
    estimated_depth_proxy_cm REAL NOT NULL,
    pothole_class TEXT NOT NULL CHECK (
        pothole_class IN ('Surface Crack', 'Moderate Cavity', 'Critical Hazard')
    ),
    estimated_repair_cost_inr REAL NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incident_reports(incident_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_incidents_dedup
    ON incident_reports (zone_id, incident_type, status, last_seen_timestamp);

CREATE INDEX IF NOT EXISTS idx_incidents_plate
    ON incident_reports (plate_number);
"""


# ==============================================================================
# 2. Initialization & Realistic CSMSS Kanchanwadi Seed Data
# ==============================================================================
def init_db(db_path: Optional[Path] = None, seed_initial_data: bool = True) -> None:
    """Create the 5 relational tables and populate default CSMSS Kanchanwadi records if empty."""
    with get_db_connection(db_path) as conn:
        conn.executescript(SCHEMA_DDL)

        # Always ensure the 4 canonical CSMSS campus zones exist
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

        if not seed_initial_data:
            return

        # Seed users if empty
        user_count = conn.execute("SELECT COUNT(*) FROM users;").fetchone()[0]
        if user_count == 0:
            seed_users = [
                ("Dr. Ulhas Shinde", "Campus Admin", "Principal Office", "+91-9422201101"),
                ("Prof. Sachin Deshmukh", "Faculty", "Computer Science & Engineering", "+91-9823014522"),
                ("Prof. Savita Kulkarni", "Faculty", "Civil Engineering", "+91-9423156789"),
                ("Rameshwar Jadhav", "Security Guard", "Main Gate Security", "+91-9765401234"),
                ("Santosh Gaikwad", "Civil Maintenance Staff", "Campus Estate & Works", "+91-9890123456"),
                ("Atharva Joshi", "Student", "Artificial Intelligence & Data Science", "+91-9158009876"),
            ]
            conn.executemany(
                """
                INSERT INTO users (full_name, role, department, contact_no)
                VALUES (?, ?, ?, ?);
                """,
                seed_users,
            )

        # Seed registered vehicles if empty (Chhatrapati Sambhajinagar MH20 RTO series)
        vehicle_count = conn.execute("SELECT COUNT(*) FROM registered_vehicles;").fetchone()[0]
        if vehicle_count == 0:
            seed_vehicles = [
                ("MH20EE4501", "Prof. Sachin Deshmukh", "Faculty", "Faculty 4W", "+91-9823014522", 1),
                ("MH20DV8821", "Prof. Savita Kulkarni", "Faculty", "Faculty 4W", "+91-9423156789", 1),
                ("MH20CS1008", "CSMSS Transport Cell", "Campus Admin", "Campus Bus", "+91-2402379011", 1),
                ("MH20AM1080", "CSMSS Medical Emergency Unit", "Campus Admin", "Emergency Ambulance", "+91-2402379108", 1),
                ("MH20BQ7742", "Atharva Joshi", "Student", "Student 2W", "+91-9158009876", 1),
                ("MH20GF9910", "Unverified Visitor / Suspended Pass", "Student", "Student 2W", "+91-9000011122", 0),
            ]
            conn.executemany(
                """
                INSERT INTO registered_vehicles (
                    plate_number, owner_name, role, vehicle_type, contact_no, is_whitelisted
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                seed_vehicles,
            )

        # Seed baseline incidents if empty
        inc_count = conn.execute("SELECT COUNT(*) FROM incident_reports;").fetchone()[0]
        if inc_count == 0:
            now_dt = datetime.now(IST)
            t1 = (now_dt - timedelta(minutes=42)).strftime("%Y-%m-%dT%H:%M:%S")
            t2 = (now_dt - timedelta(minutes=28)).strftime("%Y-%m-%dT%H:%M:%S")
            t3 = (now_dt - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")

            # Seed Incident 1: Emergency Lane Blocked at Main Gate
            cur1 = conn.execute(
                """
                INSERT INTO incident_reports (
                    timestamp, last_seen_timestamp, zone_id, incident_type,
                    plate_number, severity, status, repeat_count,
                    confidence_score, snapshot_path, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    t1,
                    t2,
                    "ZONE-01",
                    "Emergency Lane Blocked",
                    "MH20GF9910",
                    "Critical",
                    "Open",
                    3,
                    0.94,
                    "data/snapshots/zone01_emergency_block.jpg",
                    "Two-wheeler parked inside yellow cross-hatched ambulance bay at Main Gate & Security Plaza.",
                ),
            )
            _ = cur1.lastrowid

            # Seed Incident 2: Road Damage at Diploma Building Circle
            cur2 = conn.execute(
                """
                INSERT INTO incident_reports (
                    timestamp, last_seen_timestamp, zone_id, incident_type,
                    plate_number, severity, status, repeat_count,
                    confidence_score, snapshot_path, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    t2,
                    t3,
                    "ZONE-02",
                    "Road Damage",
                    None,
                    "High",
                    "Investigating",
                    2,
                    0.91,
                    "data/snapshots/zone02_pothole_cavity.jpg",
                    "Monsoon asphalt ravelling and moderate cavity detected on Diploma Building Circle rotary edge.",
                ),
            )
            road_inc_id = int(cur2.lastrowid)
            area_sqm = 0.84
            sev_idx = 68.5
            depth_proxy = 6.4
            cost_inr = compute_repair_cost_inr(
                area_sqm, sev_idx, CAMPUS_ZONES["ZONE-02"]["criticality_weight"]
            )
            conn.execute(
                """
                INSERT INTO road_damage_logs (
                    incident_id, surface_area_sqm, severity_index,
                    estimated_depth_proxy_cm, pothole_class, estimated_repair_cost_inr
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    road_inc_id,
                    area_sqm,
                    sev_idx,
                    depth_proxy,
                    "Moderate Cavity",
                    cost_inr,
                ),
            )

            # Seed Incident 3: Unlawful Parking at Central Library Approach Road
            conn.execute(
                """
                INSERT INTO incident_reports (
                    timestamp, last_seen_timestamp, zone_id, incident_type,
                    plate_number, severity, status, repeat_count,
                    confidence_score, snapshot_path, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    t3,
                    t3,
                    "ZONE-03",
                    "Unlawful Parking",
                    "MH20BQ7742",
                    "Medium",
                    "Open",
                    1,
                    0.89,
                    "data/snapshots/zone03_library_no_parking.jpg",
                    "Vehicle stationary in silent walkway no-parking buffer along Central Library Approach Road.",
                ),
            )


def reset_database(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Clear all dynamic tables and restore clean CSMSS Kanchanwadi baseline seed data."""
    with get_db_connection(db_path) as conn:
        conn.execute("DELETE FROM road_damage_logs;")
        conn.execute("DELETE FROM incident_reports;")
        conn.execute("DELETE FROM registered_vehicles;")
        conn.execute("DELETE FROM users;")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('road_damage_logs', 'incident_reports', 'registered_vehicles', 'users');")
    init_db(db_path=db_path, seed_initial_data=True)
    return {
        "status": "reset_complete",
        "timestamp": now_ist_iso(),
        "seeded_zones": len(CAMPUS_ZONES),
    }


# ==============================================================================
# 3. Query & Persistence Helpers
# ==============================================================================
def get_database_health(db_path: Optional[Path] = None) -> Tuple[str, List[str]]:
    """Return ('healthy', table_names) if all 5 required tables exist in SQLite."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
            """
        ).fetchall()
        table_names = [r["name"] for r in rows]
        required = {
            "users",
            "campus_zones",
            "registered_vehicles",
            "incident_reports",
            "road_damage_logs",
        }
        is_ok = required.issubset(set(table_names))
        return ("healthy" if is_ok else "degraded", table_names)


def list_zones(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all 4 configured CSMSS campus zones ordered by zone_id."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM campus_zones ORDER BY zone_id ASC;"
        ).fetchall()
        return [dict(r) for r in rows]


def get_zone_by_id(zone_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Fetch a single campus zone record by zone_id."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM campus_zones WHERE zone_id = ?;", (zone_id,)
        ).fetchone()
        return dict(row) if row else None


def list_vehicles(
    whitelisted_only: Optional[bool] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return registered campus vehicles."""
    with get_db_connection(db_path) as conn:
        if whitelisted_only is None:
            rows = conn.execute(
                "SELECT * FROM registered_vehicles ORDER BY vehicle_id DESC;"
            ).fetchall()
        else:
            flag = 1 if whitelisted_only else 0
            rows = conn.execute(
                "SELECT * FROM registered_vehicles WHERE is_whitelisted = ? ORDER BY vehicle_id DESC;",
                (flag,),
            ).fetchall()
        results = []
        for r in rows:
            item = dict(r)
            item["is_whitelisted"] = bool(item["is_whitelisted"])
            results.append(item)
        return results


def upsert_vehicle(
    plate_number: str,
    owner_name: str,
    role: str,
    vehicle_type: str,
    contact_no: str,
    is_whitelisted: bool = True,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Insert or update a vehicle in `registered_vehicles` by unique `plate_number`."""
    clean_plate = "".join(ch for ch in plate_number.upper() if ch.isalnum())
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO registered_vehicles (
                plate_number, owner_name, role, vehicle_type, contact_no, is_whitelisted
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(plate_number) DO UPDATE SET
                owner_name = excluded.owner_name,
                role = excluded.role,
                vehicle_type = excluded.vehicle_type,
                contact_no = excluded.contact_no,
                is_whitelisted = excluded.is_whitelisted;
            """,
            (clean_plate, owner_name, role, vehicle_type, contact_no, 1 if is_whitelisted else 0),
        )
        row = conn.execute(
            "SELECT * FROM registered_vehicles WHERE plate_number = ?;",
            (clean_plate,),
        ).fetchone()
        res = dict(row)
        res["is_whitelisted"] = bool(res["is_whitelisted"])
        return res


def get_incident_by_id(incident_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single incident joined with its zone metadata and optional road damage telemetry."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                i.*,
                z.zone_name,
                r.log_id,
                r.surface_area_sqm,
                r.severity_index,
                r.estimated_depth_proxy_cm,
                r.pothole_class,
                r.estimated_repair_cost_inr
            FROM incident_reports i
            LEFT JOIN campus_zones z ON i.zone_id = z.zone_id
            LEFT JOIN road_damage_logs r ON i.incident_id = r.incident_id
            WHERE i.incident_id = ?;
            """,
            (incident_id,),
        ).fetchone()
        if not row:
            return None
        return _format_joined_incident_row(dict(row))


def list_incidents(
    zone_id: Optional[str] = None,
    status: Optional[str] = None,
    incident_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List incidents with optional filters and embedded road damage logs."""
    clauses: List[str] = []
    params: List[Any] = []
    if zone_id:
        clauses.append("i.zone_id = ?")
        params.append(zone_id)
    if status:
        clauses.append("i.status = ?")
        params.append(status)
    if incident_type:
        clauses.append("i.incident_type = ?")
        params.append(incident_type)
    if severity:
        clauses.append("i.severity = ?")
        params.append(severity)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))

    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                i.*,
                z.zone_name,
                r.log_id,
                r.surface_area_sqm,
                r.severity_index,
                r.estimated_depth_proxy_cm,
                r.pothole_class,
                r.estimated_repair_cost_inr
            FROM incident_reports i
            LEFT JOIN campus_zones z ON i.zone_id = z.zone_id
            LEFT JOIN road_damage_logs r ON i.incident_id = r.incident_id
            {where_sql}
            ORDER BY i.last_seen_timestamp DESC, i.incident_id DESC
            LIMIT ?;
            """,
            tuple(params),
        ).fetchall()
        return [_format_joined_incident_row(dict(r)) for r in rows]


def _format_joined_incident_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert joined SQLite row into structured incident dict with nested road_damage object."""
    road_damage_obj = None
    if raw.get("log_id") is not None:
        road_damage_obj = {
            "log_id": raw["log_id"],
            "incident_id": raw["incident_id"],
            "surface_area_sqm": raw["surface_area_sqm"],
            "severity_index": raw["severity_index"],
            "estimated_depth_proxy_cm": raw["estimated_depth_proxy_cm"],
            "depth_estimation_method": "Heuristic Shadow-Intensity Proxy (Monocular CV)",
            "pothole_class": raw["pothole_class"],
            "estimated_repair_cost_inr": raw["estimated_repair_cost_inr"],
            "zone_id": raw["zone_id"],
            "zone_name": raw.get("zone_name"),
            "timestamp": raw["last_seen_timestamp"],
            "status": raw["status"],
            "repeat_count": raw["repeat_count"],
        }

    return {
        "incident_id": raw["incident_id"],
        "timestamp": raw["timestamp"],
        "last_seen_timestamp": raw["last_seen_timestamp"],
        "zone_id": raw["zone_id"],
        "zone_name": raw.get("zone_name"),
        "incident_type": raw["incident_type"],
        "plate_number": raw["plate_number"],
        "severity": raw["severity"],
        "status": raw["status"],
        "repeat_count": raw["repeat_count"],
        "confidence_score": raw["confidence_score"],
        "snapshot_path": raw["snapshot_path"],
        "details": raw["details"],
        "road_damage": road_damage_obj,
    }


def update_incident_status(
    incident_id: int,
    new_status: str,
    resolution_notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Update an incident's lifecycle status (`Open`, `Investigating`, `Resolved`)."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT details FROM incident_reports WHERE incident_id = ?;",
            (incident_id,),
        ).fetchone()
        if not row:
            return None

        updated_details = row["details"] or ""
        if resolution_notes:
            stamp = now_ist_iso()
            note_suffix = f" [Status -> {new_status} @ {stamp}: {resolution_notes.strip()}]"
            updated_details = (updated_details + note_suffix).strip()

        conn.execute(
            """
            UPDATE incident_reports
            SET status = ?, details = ?
            WHERE incident_id = ?;
            """,
            (new_status, updated_details, incident_id),
        )
    return get_incident_by_id(incident_id, db_path=db_path)


def list_road_damage_logs(
    zone_id: Optional[str] = None,
    pothole_class: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return calibrated road damage logs joined with incident and campus zone metadata."""
    clauses: List[str] = []
    params: List[Any] = []
    if zone_id:
        clauses.append("i.zone_id = ?")
        params.append(zone_id)
    if pothole_class:
        clauses.append("r.pothole_class = ?")
        params.append(pothole_class)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                r.*,
                i.zone_id,
                z.zone_name,
                i.last_seen_timestamp AS timestamp,
                i.status,
                i.repeat_count
            FROM road_damage_logs r
            JOIN incident_reports i ON r.incident_id = i.incident_id
            LEFT JOIN campus_zones z ON i.zone_id = z.zone_id
            {where_sql}
            ORDER BY r.severity_index DESC, r.log_id DESC;
            """,
            tuple(params),
        ).fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["depth_estimation_method"] = "Heuristic Shadow-Intensity Proxy (Monocular CV)"
            results.append(item)
        return results


def get_analytics_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compute comprehensive KPI counts, deduplication savings, zone breakdown, and severity distribution."""
    with get_db_connection(db_path) as conn:
        inc_stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_incidents,
                SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END) AS open_incidents,
                SUM(CASE WHEN status = 'Investigating' THEN 1 ELSE 0 END) AS investigating_incidents,
                SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) AS resolved_incidents,
                SUM(CASE WHEN severity = 'Critical' THEN 1 ELSE 0 END) AS critical_incidents,
                COALESCE(SUM(repeat_count), 0) AS total_repeat_events_captured,
                COALESCE(SUM(CASE WHEN repeat_count > 1 THEN repeat_count - 1 ELSE 0 END), 0) AS deduplication_savings_count
            FROM incident_reports;
            """
        ).fetchone()

        total_incidents = int(inc_stats["total_incidents"] or 0)
        total_events = int(inc_stats["total_repeat_events_captured"] or 0)
        dedup_savings = int(inc_stats["deduplication_savings_count"] or 0)
        dedup_efficiency = round((dedup_savings / total_events) * 100.0, 2) if total_events > 0 else 0.0

        veh_stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_vehicles,
                COALESCE(SUM(CASE WHEN is_whitelisted = 1 THEN 1 ELSE 0 END), 0) AS whitelisted_vehicles
            FROM registered_vehicles;
            """
        ).fetchone()

        road_stats = conn.execute(
            """
            SELECT
                COUNT(*) AS road_damage_count,
                ROUND(COALESCE(SUM(surface_area_sqm), 0.0), 3) AS total_damaged_area_sqm,
                ROUND(COALESCE(SUM(estimated_repair_cost_inr), 0.0), 2) AS total_estimated_repair_cost_inr
            FROM road_damage_logs;
            """
        ).fetchone()

        # Zone breakdown initialized with all 4 CSMSS zones
        zone_breakdown: Dict[str, int] = {z_id: 0 for z_id in CAMPUS_ZONES}
        for r in conn.execute(
            "SELECT zone_id, COUNT(*) AS cnt FROM incident_reports GROUP BY zone_id;"
        ).fetchall():
            zone_breakdown[r["zone_id"]] = int(r["cnt"])

        # Severity distribution initialized with all 4 levels
        severity_distribution: Dict[str, int] = {
            "Low": 0,
            "Medium": 0,
            "High": 0,
            "Critical": 0,
        }
        for r in conn.execute(
            "SELECT severity, COUNT(*) AS cnt FROM incident_reports GROUP BY severity;"
        ).fetchall():
            severity_distribution[r["severity"]] = int(r["cnt"])

        # Incident type distribution
        incident_type_distribution: Dict[str, int] = {
            "Unlawful Parking": 0,
            "Road Damage": 0,
            "Unauthorized Vehicle": 0,
            "Emergency Lane Blocked": 0,
        }
        for r in conn.execute(
            "SELECT incident_type, COUNT(*) AS cnt FROM incident_reports GROUP BY incident_type;"
        ).fetchall():
            incident_type_distribution[r["incident_type"]] = int(r["cnt"])

        return {
            "total_incidents": total_incidents,
            "open_incidents": int(inc_stats["open_incidents"] or 0),
            "investigating_incidents": int(inc_stats["investigating_incidents"] or 0),
            "resolved_incidents": int(inc_stats["resolved_incidents"] or 0),
            "critical_incidents": int(inc_stats["critical_incidents"] or 0),
            "total_repeat_events_captured": total_events,
            "deduplication_savings_count": dedup_savings,
            "deduplication_efficiency_pct": dedup_efficiency,
            "registered_vehicles_count": int(veh_stats["total_vehicles"] or 0),
            "whitelisted_vehicles_count": int(veh_stats["whitelisted_vehicles"] or 0),
            "road_damage_count": int(road_stats["road_damage_count"] or 0),
            "total_damaged_area_sqm": float(road_stats["total_damaged_area_sqm"] or 0.0),
            "total_estimated_repair_cost_inr": float(
                road_stats["total_estimated_repair_cost_inr"] or 0.0
            ),
            "zone_breakdown": zone_breakdown,
            "severity_distribution": severity_distribution,
            "incident_type_distribution": incident_type_distribution,
        }