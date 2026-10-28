"""
FastAPI Backend Service for CSMSS Kanchanwadi Civic Monitor v2.0.
CSMSS Chh. Shahu College of Engineering, Kanchanwadi, Chhatrapati Sambhajinagar.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from backend.database import (
    get_analytics_summary,
    get_database_health,
    get_incident_by_id,
    get_zone_by_id,
    init_db,
    list_incidents,
    list_road_damage_logs,
    list_vehicles,
    list_zones,
    reset_database,
    update_incident_status,
    upsert_vehicle,
)
from backend.deduplicator import record_or_deduplicate_incident
from backend.models import (
    AnalyticsSummaryResponse,
    CampusZoneRecord,
    DemoScenarioResponse,
    HealthResponse,
    IncidentCreateRequest,
    IncidentCreateResponse,
    IncidentRecord,
    IncidentStatusUpdateRequest,
    RoadDamageLogRecord,
    VehicleCreateRequest,
    VehicleRecord,
)
from config.settings import (
    APP_TITLE,
    APP_VERSION,
    DB_PATH,
    ENGINE_MODE,
    INSTITUTION_NAME,
)

SERVICE_START_MONOTONIC: float = time.monotonic()


def _as_str(val: Any) -> str:
    return val.value if isinstance(val, Enum) else str(val)


# ==============================================================================
# Preset Campus Demo Scenarios (CSMSS Kanchanwadi)
# ==============================================================================
PRESET_DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "main_gate_ambulance_block": {
        "title": "Main Gate Emergency Ambulance Lane Blockade (ZONE-01)",
        "payload": {
            "zone_id": "ZONE-01",
            "incident_type": "Emergency Lane Blocked",
            "plate_number": "MH20GF9910",
            "severity": "Critical",
            "status": "Open",
            "confidence_score": 0.96,
            "snapshot_path": "data/snapshots/demo_zone01_ambulance_block.jpg",
            "details": (
                "Unwhitelisted two-wheeler MH20GF9910 obstructing Main Gate emergency ambulance "
                "apron during peak morning arrival."
            ),
            "road_damage": None,
        },
    },
    "diploma_circle_pothole_hazard": {
        "title": "Diploma Building Circle Critical Pothole Hazard (ZONE-02)",
        "payload": {
            "zone_id": "ZONE-02",
            "incident_type": "Road Damage",
            "plate_number": None,
            "severity": "Critical",
            "status": "Open",
            "confidence_score": 0.93,
            "snapshot_path": "data/snapshots/demo_zone02_critical_pothole.jpg",
            "details": (
                "Deep waterlogged asphalt cavity detected at Diploma Building Circle bus turning radius; "
                "high two-wheeler skidding risk."
            ),
            "road_damage": {
                "surface_area_sqm": 1.45,
                "severity_index": 88.0,
                "estimated_depth_proxy_cm": 11.2,
                "pothole_class": "Critical Hazard",
                "estimated_repair_cost_inr": None,
            },
        },
    },
    "library_road_unlawful_parking": {
        "title": "Central Library Approach Road Unlawful Parking (ZONE-03)",
        "payload": {
            "zone_id": "ZONE-03",
            "incident_type": "Unlawful Parking",
            "plate_number": "MH20BQ7742",
            "severity": "Medium",
            "status": "Open",
            "confidence_score": 0.90,
            "snapshot_path": "data/snapshots/demo_zone03_unlawful_parking.jpg",
            "details": (
                "Student two-wheeler MH20BQ7742 parked inside pedestrian walkway buffer "
                "outside Central Library reading hall."
            ),
            "road_damage": None,
        },
    },
    "workshop_bay_unauthorized_entry": {
        "title": "Mechanical Workshop Bay Unauthorized Commercial Vehicle (ZONE-04)",
        "payload": {
            "zone_id": "ZONE-04",
            "incident_type": "Unauthorized Vehicle",
            "plate_number": "MH20CT6105",
            "severity": "High",
            "status": "Open",
            "confidence_score": 0.92,
            "snapshot_path": "data/snapshots/demo_zone04_unauthorized_truck.jpg",
            "details": (
                "Unregistered external vehicle MH20CT6105 detected inside Mechanical Workshop Bay "
                "restricted service yard without gate pass."
            ),
            "road_damage": None,
        },
    },
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize SQLite schema and CSMSS Kanchanwadi baseline seed data on startup."""
    init_db(seed_initial_data=True)
    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        f"Backend REST API & Temporal Deduplication Engine for {INSTITUTION_NAME}. "
        "Supports campus zone telemetry, vehicle whitelist enforcement, and calibrated "
        "road damage assessment."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 1. Health & Analytics Endpoints
# ==============================================================================
@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Return SQLite database status, active CV/DL engine mode, total tables, and service uptime."""
    db_status, table_names = get_database_health()
    uptime_sec = round(time.monotonic() - SERVICE_START_MONOTONIC, 2)
    return HealthResponse(
        status="ok" if db_status == "healthy" else "degraded",
        app_title=APP_TITLE,
        version=APP_VERSION,
        db_status=db_status,
        db_path=str(DB_PATH),
        engine_mode=ENGINE_MODE,
        total_tables=len(table_names),
        table_names=table_names,
        uptime_seconds=uptime_sec,
    )


@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse, tags=["Analytics"])
def get_analytics() -> AnalyticsSummaryResponse:
    """Return KPI counts, deduplication savings count, zone breakdown, and severity distribution."""
    summary_data = get_analytics_summary()
    return AnalyticsSummaryResponse(**summary_data)


# ==============================================================================
# 2. Campus Zones & Incidents Endpoints
# ==============================================================================
@app.get("/zones", response_model=List[CampusZoneRecord], tags=["Zones"])
def get_zones() -> List[CampusZoneRecord]:
    """Return all 4 configured CSMSS Kanchanwadi campus monitoring zones."""
    zones = list_zones()
    return [CampusZoneRecord(**z) for z in zones]


@app.get("/incidents", response_model=List[IncidentRecord], tags=["Incidents"])
def get_incidents(
    zone_id: Optional[str] = Query(default=None, description="Filter by zone_id (ZONE-01..ZONE-04)"),
    status_filter: Optional[str] = Query(
        default=None, alias="status", description="Filter by status (Open, Investigating, Resolved)"
    ),
    incident_type: Optional[str] = Query(default=None, description="Filter by incident_type"),
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[IncidentRecord]:
    """List campus civic incidents ordered by most recently seen."""
    rows = list_incidents(
        zone_id=zone_id,
        status=status_filter,
        incident_type=incident_type,
        severity=severity,
        limit=limit,
    )
    return [IncidentRecord(**r) for r in rows]


@app.post(
    "/incidents",
    response_model=IncidentCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Incidents"],
)
def create_incident(payload: IncidentCreateRequest) -> IncidentCreateResponse:
    """
    Report a new campus incident with automatic 15-minute temporal deduplication.
    If an active matching incident exists in the same zone within `DEDUP_WINDOW_MINUTES`,
    `repeat_count` is incremented and `last_seen_timestamp` + `confidence_score` are updated.
    """
    zone = get_zone_by_id(payload.zone_id)
    if not zone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown campus zone_id: {payload.zone_id}",
        )

    road_dict = payload.road_damage.to_dict() if payload.road_damage else None
    result = record_or_deduplicate_incident(
        zone_id=payload.zone_id,
        incident_type=_as_str(payload.incident_type),
        severity=_as_str(payload.severity),
        confidence_score=payload.confidence_score,
        plate_number=payload.plate_number,
        status=_as_str(payload.status),
        snapshot_path=payload.snapshot_path,
        details=payload.details,
        road_damage=road_dict,
    )
    return IncidentCreateResponse(**result)


@app.patch(
    "/incidents/{incident_id}/status",
    response_model=IncidentRecord,
    tags=["Incidents"],
)
def patch_incident_status(
    incident_id: int,
    payload: IncidentStatusUpdateRequest,
) -> IncidentRecord:
    """Transition an incident between `Open`, `Investigating`, and `Resolved`."""
    existing = get_incident_by_id(incident_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident #{incident_id} not found.",
        )
    updated = update_incident_status(
        incident_id=incident_id,
        new_status=_as_str(payload.status),
        resolution_notes=payload.resolution_notes,
    )
    return IncidentRecord(**updated)


# ==============================================================================
# 3. Registered Vehicles & Road Damage Endpoints
# ==============================================================================
@app.get("/vehicles", response_model=List[VehicleRecord], tags=["Vehicles"])
def get_vehicles(
    whitelisted_only: Optional[bool] = Query(
        default=None, description="Optional filter by whitelist authorization status"
    ),
) -> List[VehicleRecord]:
    """List registered student, faculty, bus, and emergency vehicles."""
    rows = list_vehicles(whitelisted_only=whitelisted_only)
    return [VehicleRecord(**r) for r in rows]


@app.post(
    "/vehicles",
    response_model=VehicleRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["Vehicles"],
)
def register_vehicle(payload: VehicleCreateRequest) -> VehicleRecord:
    """Register or update a campus vehicle in `registered_vehicles`."""
    record = upsert_vehicle(
        plate_number=payload.plate_number,
        owner_name=payload.owner_name,
        role=_as_str(payload.role),
        vehicle_type=_as_str(payload.vehicle_type),
        contact_no=payload.contact_no,
        is_whitelisted=payload.is_whitelisted,
    )
    return VehicleRecord(**record)


@app.get("/road-damage", response_model=List[RoadDamageLogRecord], tags=["Road Damage"])
def get_road_damage(
    zone_id: Optional[str] = Query(default=None, description="Filter by campus zone_id"),
    pothole_class: Optional[str] = Query(
        default=None,
        description="Filter by pothole_class (Surface Crack, Moderate Cavity, Critical Hazard)",
    ),
) -> List[RoadDamageLogRecord]:
    """Return calibrated road damage telemetry logs with shadow-intensity depth proxies and repair costs."""
    logs = list_road_damage_logs(zone_id=zone_id, pothole_class=pothole_class)
    return [RoadDamageLogRecord(**log) for log in logs]


# ==============================================================================
# 4. Preset Demo Scenario Trigger & Reset Endpoint
# ==============================================================================
SCENARIO_ALIASES: Dict[str, str] = {
    "1": "main_gate_ambulance_block",
    "scenario_1": "main_gate_ambulance_block",
    "zone_01": "main_gate_ambulance_block",
    "main_gate_ambulance_block": "main_gate_ambulance_block",
    "2": "diploma_circle_pothole_hazard",
    "scenario_2": "diploma_circle_pothole_hazard",
    "zone_02": "diploma_circle_pothole_hazard",
    "diploma_circle_pothole_hazard": "diploma_circle_pothole_hazard",
    "3": "library_road_unlawful_parking",
    "scenario_3": "library_road_unlawful_parking",
    "zone_03": "library_road_unlawful_parking",
    "library_road_unlawful_parking": "library_road_unlawful_parking",
    "4": "workshop_bay_unauthorized_entry",
    "scenario_4": "workshop_bay_unauthorized_entry",
    "zone_04": "workshop_bay_unauthorized_entry",
    "workshop_bay_unauthorized_entry": "workshop_bay_unauthorized_entry",
}


@app.post("/demo/trigger/{scenario_key}", response_model=DemoScenarioResponse, tags=["Demo"])
def trigger_demo_scenario(scenario_key: str) -> DemoScenarioResponse:
    """
    Trigger one of the 4 preset CSMSS Kanchanwadi demonstration scenarios or reset the database:
    - `main_gate_ambulance_block` (or `1` / `scenario_1` / `zone_01`): Emergency Lane Blocked at Main Gate
    - `diploma_circle_pothole_hazard` (or `2` / `scenario_2` / `zone_02`): Critical Pothole Hazard at Diploma Circle
    - `library_road_unlawful_parking` (or `3` / `scenario_3` / `zone_03`): Unlawful Parking on Central Library Road
    - `workshop_bay_unauthorized_entry` (or `4` / `scenario_4` / `zone_04`): Unauthorized Vehicle at Mechanical Workshop Bay
    - `reset` (or `reset_db`): Restore database to clean baseline seed state
    """
    normalized_key = scenario_key.strip().lower()
    if normalized_key in {"reset", "reset_db", "clear"}:
        reset_info = reset_database()
        return DemoScenarioResponse(
            scenario_key="reset",
            action="database_reset",
            summary="Database reset to initial CSMSS Kanchanwadi baseline seed data.",
            result=reset_info,
        )

    canonical_key = SCENARIO_ALIASES.get(normalized_key)
    if not canonical_key or canonical_key not in PRESET_DEMO_SCENARIOS:
        valid_keys = list(PRESET_DEMO_SCENARIOS.keys()) + ["reset"]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown scenario_key '{scenario_key}'. Valid options: {valid_keys}",
        )

    scenario = PRESET_DEMO_SCENARIOS[canonical_key]
    p = scenario["payload"]
    outcome = record_or_deduplicate_incident(
        zone_id=p["zone_id"],
        incident_type=p["incident_type"],
        severity=p["severity"],
        confidence_score=p["confidence_score"],
        plate_number=p["plate_number"],
        status=p["status"],
        snapshot_path=p["snapshot_path"],
        details=p["details"],
        road_damage=p["road_damage"],
    )
    return DemoScenarioResponse(
        scenario_key=canonical_key,
        action="scenario_triggered",
        summary=scenario["title"],
        result=outcome,
    )