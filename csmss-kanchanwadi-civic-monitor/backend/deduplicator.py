"""
Temporal Deduplication Engine for CSMSS Kanchanwadi Civic Monitor v2.0.

Prevents duplicate incident spam when a camera or patrol feed repeatedly detects
the same stationary vehicle or road hazard within `DEDUP_WINDOW_MINUTES` (default 15 mins).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from backend.database import (
    IST,
    compute_repair_cost_inr,
    get_db_connection,
    get_incident_by_id,
    now_ist_iso,
)
from config.settings import CAMPUS_ZONES, DEDUP_WINDOW_MINUTES

SEVERITY_RANK: Dict[str, int] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}


def _parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO-8601 timestamp string into timezone-aware IST datetime."""
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        return dt
    except (ValueError, TypeError):
        return None


def find_active_duplicate(
    zone_id: str,
    incident_type: str,
    plate_number: Optional[str] = None,
    pothole_class: Optional[str] = None,
    dedup_window_minutes: int = DEDUP_WINDOW_MINUTES,
    reference_time: Optional[datetime] = None,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Search for an `Open` or `Investigating` incident in the same `zone_id` and `incident_type`
    whose `last_seen_timestamp` falls within `dedup_window_minutes`.

    Matching Rules:
    1. Vehicle incidents (`Unlawful Parking`, `Unauthorized Vehicle`, `Emergency Lane Blocked`):
       Matches on (`zone_id`, `incident_type`, normalized `plate_number`). If `plate_number`
       is None/unreadable, matches an active unplated incident of the same type in that zone.
    2. Road damage incidents (`Road Damage`):
       Matches on (`zone_id`, `incident_type == 'Road Damage'`) and matching `pothole_class`
       in `road_damage_logs` (or any active `Road Damage` in the same zone if `pothole_class` is omitted).
    """
    now_dt = reference_time or datetime.now(IST)
    cutoff_dt = now_dt - timedelta(minutes=dedup_window_minutes)
    clean_plate = (
        "".join(ch for ch in plate_number.upper() if ch.isalnum())
        if plate_number
        else None
    )

    with get_db_connection(db_path) as conn:
        if incident_type == "Road Damage":
            if pothole_class:
                rows = conn.execute(
                    """
                    SELECT i.*, r.pothole_class
                    FROM incident_reports i
                    LEFT JOIN road_damage_logs r ON i.incident_id = r.incident_id
                    WHERE i.zone_id = ?
                      AND i.incident_type = 'Road Damage'
                      AND i.status IN ('Open', 'Investigating')
                      AND (r.pothole_class = ? OR r.pothole_class IS NULL)
                    ORDER BY i.last_seen_timestamp DESC, i.incident_id DESC;
                    """,
                    (zone_id, pothole_class),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT i.*
                    FROM incident_reports i
                    WHERE i.zone_id = ?
                      AND i.incident_type = 'Road Damage'
                      AND i.status IN ('Open', 'Investigating')
                    ORDER BY i.last_seen_timestamp DESC, i.incident_id DESC;
                    """,
                    (zone_id,),
                ).fetchall()
        else:
            if clean_plate:
                rows = conn.execute(
                    """
                    SELECT i.*
                    FROM incident_reports i
                    WHERE i.zone_id = ?
                      AND i.incident_type = ?
                      AND i.status IN ('Open', 'Investigating')
                      AND UPPER(REPLACE(REPLACE(COALESCE(i.plate_number, ''), ' ', ''), '-', '')) = ?
                    ORDER BY i.last_seen_timestamp DESC, i.incident_id DESC;
                    """,
                    (zone_id, incident_type, clean_plate),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT i.*
                    FROM incident_reports i
                    WHERE i.zone_id = ?
                      AND i.incident_type = ?
                      AND i.status IN ('Open', 'Investigating')
                      AND (i.plate_number IS NULL OR i.plate_number = '')
                    ORDER BY i.last_seen_timestamp DESC, i.incident_id DESC;
                    """,
                    (zone_id, incident_type),
                ).fetchall()

        for row in rows:
            last_seen_dt = _parse_iso_timestamp(row["last_seen_timestamp"])
            if last_seen_dt and last_seen_dt >= cutoff_dt:
                return dict(row)

    return None


def record_or_deduplicate_incident(
    zone_id: str,
    incident_type: str,
    severity: str,
    confidence_score: float = 0.88,
    plate_number: Optional[str] = None,
    status: str = "Open",
    snapshot_path: Optional[str] = None,
    details: str = "",
    road_damage: Optional[Dict[str, Any]] = None,
    dedup_window_minutes: int = DEDUP_WINDOW_MINUTES,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Core Temporal Deduplication Entrypoint:
    - Checks if an `Open` or `Investigating` incident with the same `zone_id`,
      `incident_type`, and `plate_number` (or `pothole_class` in the same zone)
      occurred within `DEDUP_WINDOW_MINUTES` (default 15 minutes).
    - If found: increments `repeat_count`, updates `last_seen_timestamp` and
      `confidence_score` (keeping the higher confidence & escalated severity if applicable),
      updates road damage metrics if provided, and returns:
      `{"deduplicated": True, "incident_id": existing_id, "repeat_count": new_count, ...}`
    - Otherwise: inserts a new row in `incident_reports` (and `road_damage_logs` if applicable)
      and returns `{"deduplicated": False, "incident_id": new_id, "repeat_count": 1, ...}`.
    """
    now_dt = datetime.now(IST)
    now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%S")
    clean_plate = (
        "".join(ch for ch in plate_number.upper() if ch.isalnum())
        if plate_number
        else None
    )
    pothole_class = road_damage.get("pothole_class") if road_damage else None

    existing = find_active_duplicate(
        zone_id=zone_id,
        incident_type=incident_type,
        plate_number=clean_plate,
        pothole_class=pothole_class,
        dedup_window_minutes=dedup_window_minutes,
        reference_time=now_dt,
        db_path=db_path,
    )

    if existing is not None:
        existing_id = int(existing["incident_id"])
        new_count = int(existing["repeat_count"]) + 1
        updated_confidence = round(
            max(float(existing["confidence_score"]), float(confidence_score)), 4
        )
        existing_sev = existing["severity"]
        merged_severity = (
            severity
            if SEVERITY_RANK.get(severity, 1) > SEVERITY_RANK.get(existing_sev, 1)
            else existing_sev
        )
        updated_snapshot = snapshot_path or existing.get("snapshot_path")
        updated_details = existing.get("details") or details

        with get_db_connection(db_path) as conn:
            conn.execute(
                """
                UPDATE incident_reports
                SET repeat_count = ?,
                    last_seen_timestamp = ?,
                    confidence_score = ?,
                    severity = ?,
                    snapshot_path = ?
                WHERE incident_id = ?;
                """,
                (
                    new_count,
                    now_iso,
                    updated_confidence,
                    merged_severity,
                    updated_snapshot,
                    existing_id,
                ),
            )

            if incident_type == "Road Damage" and road_damage is not None:
                zone_weight = CAMPUS_ZONES.get(zone_id, {}).get("criticality_weight", 1.2)
                area_sqm = float(road_damage["surface_area_sqm"])
                sev_idx = float(road_damage["severity_index"])
                depth_cm = float(road_damage["estimated_depth_proxy_cm"])
                p_class = str(road_damage["pothole_class"])
                repair_cost = (
                    float(road_damage["estimated_repair_cost_inr"])
                    if road_damage.get("estimated_repair_cost_inr") is not None
                    else compute_repair_cost_inr(area_sqm, sev_idx, zone_weight)
                )
                conn.execute(
                    """
                    INSERT INTO road_damage_logs (
                        incident_id, surface_area_sqm, severity_index,
                        estimated_depth_proxy_cm, pothole_class, estimated_repair_cost_inr
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(incident_id) DO UPDATE SET
                        surface_area_sqm = MAX(road_damage_logs.surface_area_sqm, excluded.surface_area_sqm),
                        severity_index = MAX(road_damage_logs.severity_index, excluded.severity_index),
                        estimated_depth_proxy_cm = MAX(road_damage_logs.estimated_depth_proxy_cm, excluded.estimated_depth_proxy_cm),
                        pothole_class = excluded.pothole_class,
                        estimated_repair_cost_inr = MAX(road_damage_logs.estimated_repair_cost_inr, excluded.estimated_repair_cost_inr);
                    """,
                    (existing_id, area_sqm, sev_idx, depth_cm, p_class, repair_cost),
                )

        incident_record = get_incident_by_id(existing_id, db_path=db_path)
        return {
            "deduplicated": True,
            "incident_id": existing_id,
            "repeat_count": new_count,
            "message": (
                f"Temporal deduplication matched active incident #{existing_id} in {zone_id} "
                f"(within {dedup_window_minutes}m window); repeat_count incremented to {new_count}."
            ),
            "incident": incident_record,
        }

    # Otherwise insert a fresh incident row
    with get_db_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO incident_reports (
                timestamp, last_seen_timestamp, zone_id, incident_type,
                plate_number, severity, status, repeat_count,
                confidence_score, snapshot_path, details
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?);
            """,
            (
                now_iso,
                now_iso,
                zone_id,
                incident_type,
                clean_plate,
                severity,
                status,
                round(float(confidence_score), 4),
                snapshot_path,
                details,
            ),
        )
        new_id = int(cur.lastrowid)

        if incident_type == "Road Damage" and road_damage is not None:
            zone_weight = CAMPUS_ZONES.get(zone_id, {}).get("criticality_weight", 1.2)
            area_sqm = float(road_damage["surface_area_sqm"])
            sev_idx = float(road_damage["severity_index"])
            depth_cm = float(road_damage["estimated_depth_proxy_cm"])
            p_class = str(road_damage["pothole_class"])
            repair_cost = (
                float(road_damage["estimated_repair_cost_inr"])
                if road_damage.get("estimated_repair_cost_inr") is not None
                else compute_repair_cost_inr(area_sqm, sev_idx, zone_weight)
            )
            conn.execute(
                """
                INSERT INTO road_damage_logs (
                    incident_id, surface_area_sqm, severity_index,
                    estimated_depth_proxy_cm, pothole_class, estimated_repair_cost_inr
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (new_id, area_sqm, sev_idx, depth_cm, p_class, repair_cost),
            )

    incident_record = get_incident_by_id(new_id, db_path=db_path)
    return {
        "deduplicated": False,
        "incident_id": new_id,
        "repeat_count": 1,
        "message": f"Created new incident #{new_id} in {zone_id} ({incident_type}).",
        "incident": incident_record,
    }