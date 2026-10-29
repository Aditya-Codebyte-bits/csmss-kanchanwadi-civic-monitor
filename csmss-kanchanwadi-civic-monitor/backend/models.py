"""
Pydantic Domain Schemas & Validation Models for CSMSS Kanchanwadi Civic Monitor v2.0.
Supports Pydantic v2 (`field_validator`, `ConfigDict`, `model_dump`) with seamless
fallback compatibility for Pydantic v1 environments.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import pydantic
from pydantic import BaseModel, Field

PYDANTIC_V2: bool = int(pydantic.__version__.split(".")[0]) >= 2

if PYDANTIC_V2:
    from pydantic import ConfigDict, field_validator
else:
    from pydantic import validator as _v1_validator


def _normalize_plate_str(v: str) -> str:
    cleaned = "".join(ch for ch in v.upper() if ch.isalnum())
    if len(cleaned) < 4:
        raise ValueError("plate_number must contain at least 4 alphanumeric characters")
    return cleaned


def _clean_optional_plate_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    cleaned = "".join(ch for ch in v.upper() if ch.isalnum())
    return cleaned if cleaned else None


class CompatBaseModel(BaseModel):
    """Base model providing unified `.to_dict()` across Pydantic v2 (`model_dump`) and v1 (`dict`)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True, use_enum_values=True)
    else:
        class Config:
            orm_mode = True
            use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            raw = self.model_dump()
        else:
            raw = self.dict()
        return {
            k: (v.value if isinstance(v, Enum) else v)
            for k, v in raw.items()
        }


# ==============================================================================
# 1. Domain Enumerations
# ==============================================================================
class UserRole(str, Enum):
    STUDENT = "Student"
    FACULTY = "Faculty"
    SECURITY_GUARD = "Security Guard"
    CIVIL_MAINTENANCE_STAFF = "Civil Maintenance Staff"
    CAMPUS_ADMIN = "Campus Admin"


class VehicleType(str, Enum):
    STUDENT_2W = "Student 2W"
    FACULTY_4W = "Faculty 4W"
    CAMPUS_BUS = "Campus Bus"
    EMERGENCY_AMBULANCE = "Emergency Ambulance"


class IncidentType(str, Enum):
    UNLAWFUL_PARKING = "Unlawful Parking"
    ROAD_DAMAGE = "Road Damage"
    UNAUTHORIZED_VEHICLE = "Unauthorized Vehicle"
    EMERGENCY_LANE_BLOCKED = "Emergency Lane Blocked"


class SeverityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class IncidentStatus(str, Enum):
    OPEN = "Open"
    INVESTIGATING = "Investigating"
    RESOLVED = "Resolved"


class PotholeClass(str, Enum):
    SURFACE_CRACK = "Surface Crack"
    MODERATE_CAVITY = "Moderate Cavity"
    CRITICAL_HAZARD = "Critical Hazard"


# ==============================================================================
# 2. User & Zone Models
# ==============================================================================
class UserRecord(CompatBaseModel):
    user_id: int
    full_name: str
    role: UserRole
    department: str
    contact_no: str


class CampusZoneRecord(CompatBaseModel):
    zone_id: str = Field(..., description="Unique campus zone identifier (ZONE-01 to ZONE-04)")
    zone_name: str
    latitude: float
    longitude: float
    criticality_weight: float = Field(..., ge=1.0, le=2.0)
    roi_coordinates_json: str = Field(..., description="JSON serialized list of normalized [x, y] ROI points")
    px_per_meter: float = Field(..., gt=0.0, description="Homography scale in pixels per physical meter")


# ==============================================================================
# 3. Vehicle Registry Models
# ==============================================================================
class VehicleCreateRequest(CompatBaseModel):
    plate_number: str = Field(
        ...,
        min_length=4,
        max_length=20,
        description="Standardized Indian license plate, e.g. MH20EE4501",
    )
    owner_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole
    vehicle_type: VehicleType
    contact_no: str = Field(..., min_length=10, max_length=16)
    is_whitelisted: bool = Field(default=True)

    if PYDANTIC_V2:
        @field_validator("plate_number")
        @classmethod
        def normalize_plate(cls, v: str) -> str:
            return _normalize_plate_str(v)
    else:
        @_v1_validator("plate_number")
        def normalize_plate(cls, v: str) -> str:
            return _normalize_plate_str(v)


class VehicleRecord(CompatBaseModel):
    vehicle_id: int
    plate_number: str
    owner_name: str
    role: UserRole
    vehicle_type: VehicleType
    contact_no: str
    is_whitelisted: bool


# ==============================================================================
# 4. Road Damage Telemetry Models
# ==============================================================================
class RoadDamageCreatePayload(CompatBaseModel):
    surface_area_sqm: float = Field(..., gt=0.0, le=50.0, description="Calibrated damaged road area in square meters")
    severity_index: float = Field(..., ge=0.0, le=100.0, description="Composite 0-100 road damage severity score")
    estimated_depth_proxy_cm: float = Field(
        ...,
        ge=0.1,
        le=35.0,
        description="Heuristic shadow-intensity proxy for cavity depth in cm (non-LiDAR monocular estimate)",
    )
    pothole_class: PotholeClass
    estimated_repair_cost_inr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated civil patch repair cost in INR; auto-calculated if omitted",
    )


class RoadDamageLogRecord(CompatBaseModel):
    log_id: int
    incident_id: int
    surface_area_sqm: float
    severity_index: float
    estimated_depth_proxy_cm: float = Field(
        ...,
        description="Heuristic shadow-intensity proxy (cm)",
    )
    depth_estimation_method: str = Field(
        default="Heuristic Shadow-Intensity Proxy (Monocular CV)",
        description="Explicit label clarifying depth is a visual shadow-intensity proxy",
    )
    pothole_class: PotholeClass
    estimated_repair_cost_inr: float
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    timestamp: Optional[str] = None
    status: Optional[IncidentStatus] = None
    repeat_count: Optional[int] = None


# ==============================================================================
# 5. Incident Report Models
# ==============================================================================
class IncidentCreateRequest(CompatBaseModel):
    zone_id: str = Field(..., description="Target CSMSS campus zone ID (ZONE-01 to ZONE-04)")
    incident_type: IncidentType
    plate_number: Optional[str] = Field(default=None, max_length=20)
    severity: SeverityLevel
    status: IncidentStatus = Field(default=IncidentStatus.OPEN)
    confidence_score: float = Field(default=0.88, ge=0.0, le=1.0)
    snapshot_path: Optional[str] = Field(default=None)
    details: str = Field(default="", max_length=1000)
    road_damage: Optional[RoadDamageCreatePayload] = Field(
        default=None,
        description="Optional calibrated road damage telemetry when incident_type == 'Road Damage'",
    )

    if PYDANTIC_V2:
        @field_validator("plate_number")
        @classmethod
        def clean_optional_plate(cls, v: Optional[str]) -> Optional[str]:
            return _clean_optional_plate_str(v)
    else:
        @_v1_validator("plate_number")
        def clean_optional_plate(cls, v: Optional[str]) -> Optional[str]:
            return _clean_optional_plate_str(v)


class IncidentStatusUpdateRequest(CompatBaseModel):
    status: IncidentStatus
    resolution_notes: Optional[str] = Field(default=None, max_length=500)


class IncidentRecord(CompatBaseModel):
    incident_id: int
    timestamp: str
    last_seen_timestamp: str
    zone_id: str
    zone_name: Optional[str] = None
    incident_type: IncidentType
    plate_number: Optional[str] = None
    severity: SeverityLevel
    status: IncidentStatus
    repeat_count: int = Field(default=1, ge=1)
    confidence_score: float
    snapshot_path: Optional[str] = None
    details: str
    road_damage: Optional[RoadDamageLogRecord] = None


class IncidentCreateResponse(CompatBaseModel):
    deduplicated: bool
    incident_id: int
    repeat_count: int
    message: str
    incident: IncidentRecord


# ==============================================================================
# 6. Health, Analytics & Demo Models
# ==============================================================================
class HealthResponse(CompatBaseModel):
    status: str
    app_title: str
    version: str
    db_status: str
    db_path: str
    engine_mode: str
    total_tables: int
    table_names: List[str]
    uptime_seconds: float


class AnalyticsSummaryResponse(CompatBaseModel):
    total_incidents: int
    open_incidents: int
    investigating_incidents: int
    resolved_incidents: int
    critical_incidents: int
    total_repeat_events_captured: int
    deduplication_savings_count: int
    deduplication_efficiency_pct: float
    registered_vehicles_count: int
    whitelisted_vehicles_count: int
    road_damage_count: int
    total_damaged_area_sqm: float
    total_estimated_repair_cost_inr: float
    zone_breakdown: Dict[str, int]
    severity_distribution: Dict[str, int]
    incident_type_distribution: Dict[str, int]


class DemoScenarioResponse(CompatBaseModel):
    scenario_key: str
    action: str
    summary: str
    result: Dict[str, Any]