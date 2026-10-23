"""
CSMSS Kanchanwadi Civic & Security Monitoring System — REST API Backend
Engineered by: Aditya Narendra Bhalerao
Institution: CSMSS Chh. Shahu College of Engineering, Kanchanwadi, Chhatrapati Sambhajinagar

Version: 2.1.0 (High-Priority Upgrade: JWT Auth, RBAC, SQLite WAL Concurrency, Input Validation)
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration & Directory Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CSMSS_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
DB_PATH = DATA_DIR / "civic_monitor.db"

JWT_SECRET_KEY = os.environ.get("CSMSS_JWT_SECRET", "csmss_kanchanwadi_civic_sec_key_2026_aditya_bhalerao")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
MAX_FILE_SIZE_MB = 10
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cryptographic Helpers (Zero External Dependency JWT & PBKDF2 Hashing)
# ---------------------------------------------------------------------------
def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def _b64_decode(data: str) -> bytes:
    padding = "=" * (4 - (len(data) % 4)) if len(data) % 4 != 0 else ""
    return base64.urlsafe_b64decode(data + padding)

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    key = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100000)
    return key.hex(), salt

def verify_password(plain_password: str, password_hash: str, salt: str) -> bool:
    computed_hash, _ = hash_password(plain_password, salt)
    return secrets.compare_digest(computed_hash, password_hash)

def create_jwt_token(payload: dict) -> str:
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    header_b64 = _b64_encode(json.dumps(header).encode("utf-8"))
    
    body = payload.copy()
    body["exp"] = int(time.time()) + (JWT_EXPIRATION_HOURS * 3600)
    body["iat"] = int(time.time())
    body_b64 = _b64_encode(json.dumps(body).encode("utf-8"))
    
    signing_input = f"{header_b64}.{body_b64}".encode("utf-8")
    signature = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64_encode(signature)
    
    return f"{header_b64}.{body_b64}.{sig_b64}"

def decode_jwt_token(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token structure")
    
    header_b64, body_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{body_b64}".encode("utf-8")
    expected_sig = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    
    if not secrets.compare_digest(_b64_encode(expected_sig), sig_b64):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")
    
    body_json = _b64_decode(body_b64).decode("utf-8")
    payload = json.loads(body_json)
    
    if payload.get("exp", 0) < time.time():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    
    return payload

# ---------------------------------------------------------------------------
# Database & High-Concurrency Connection Manager (WAL Mode)
# ---------------------------------------------------------------------------
@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")  # High-concurrency WAL mode
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
    except sqlite3.OperationalError:
        # Graceful fallback for filesystems without WAL shared memory support
        try:
            conn.execute("PRAGMA journal_mode = DELETE;")
        except Exception:
            pass
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_database() -> None:
    with get_db_connection() as conn:
        # 1. Users Table with RBAC
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'security', 'student', 'faculty')),
                department TEXT DEFAULT 'General',
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );
        """)

        # 2. Campus Zones Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS campus_zones (
                zone_id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_code TEXT UNIQUE NOT NULL,
                zone_name TEXT NOT NULL,
                zone_type TEXT NOT NULL,
                speed_limit_kmh INTEGER DEFAULT 20,
                is_restricted INTEGER DEFAULT 0,
                description TEXT
            );
        """)

        # 3. Registered Vehicles (Gate ANPR)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registered_vehicles (
                plate_number TEXT PRIMARY KEY,
                owner_name TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                department TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                registered_at TEXT NOT NULL
            );
        """)

        # 4. Civic Incidents Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS civic_incidents (
                incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_type TEXT NOT NULL,
                zone_id INTEGER NOT NULL,
                reported_by_user_id INTEGER,
                license_plate_detected TEXT,
                is_registered_vehicle INTEGER DEFAULT 0,
                violation_confidence REAL DEFAULT 0.0,
                severity_level TEXT NOT NULL CHECK(severity_level IN ('Low', 'Medium', 'High', 'Critical')),
                status TEXT NOT NULL DEFAULT 'Open' CHECK(status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
                description TEXT,
                confidence_score REAL DEFAULT 0.0,
                image_path TEXT,
                annotated_image_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (zone_id) REFERENCES campus_zones(zone_id),
                FOREIGN KEY (reported_by_user_id) REFERENCES users(user_id)
            );
        """)

        # 5. Security Logs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS security_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                zone_id INTEGER,
                user_id INTEGER,
                details TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (zone_id) REFERENCES campus_zones(zone_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """)

        # 6. High-Frequency Query B-Tree Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON civic_incidents(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_type ON civic_incidents(incident_type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_severity ON civic_incidents(severity_level);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created ON civic_incidents(created_at);")

        # Seed initial zones if empty
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM campus_zones;")
        if cursor.fetchone()[0] == 0:
            default_zones = [
                ("ZONE-01", "Workshop Fire Exit & Emergency Lane", "Emergency Lane", 15, 1, "Zero-parking emergency corridor behind Mechanical Workshop"),
                ("ZONE-02", "CSMSS Hospital Link Road", "Ambulance Path", 20, 1, "Access corridor to CSMSS Ayurvedic Hospital & Dental Wing"),
                ("ZONE-03", "Main Academic Quadrangle Walkway", "Pedestrian Zone", 10, 1, "Pedestrian plaza strictly closed to vehicular traffic"),
                ("ZONE-04", "Central Library Parking Bay", "Faculty Bay", 20, 0, "Designated faculty & staff parking sector"),
                ("ZONE-05", "Campus Main Security Gate 1", "Checkpoint", 15, 0, "Kanchanwadi Main Entrance & ANPR Checkpoint")
            ]
            conn.executemany("""
                INSERT INTO campus_zones (zone_code, zone_name, zone_type, speed_limit_kmh, is_restricted, description)
                VALUES (?, ?, ?, ?, ?, ?);
            """, default_zones)

        # Seed default administrative & security users if empty
        cursor.execute("SELECT COUNT(*) FROM users;")
        if cursor.fetchone()[0] == 0:
            now_iso = datetime.now(timezone.utc).isoformat()
            
            # Admin: Aditya Narendra Bhalerao / Admin@CSMSS2026
            admin_pwd, admin_salt = hash_password("Admin@CSMSS2026")
            # Security: Guard Gate 1 / Guard@CSMSS2026
            sec_pwd, sec_salt = hash_password("Guard@CSMSS2026")
            # Student: Student Demo / Student@CSMSS2026
            stu_pwd, stu_salt = hash_password("Student@CSMSS2026")

            seed_users = [
                ("aditya.admin", admin_pwd, admin_salt, "Aditya Narendra Bhalerao", "admin", "Engineering & Estate Administration", now_iso),
                ("guard_gate1", sec_pwd, sec_salt, "CSMSS Security Gate 1", "security", "Campus Security Taskforce", now_iso),
                ("student_demo", stu_pwd, stu_salt, "CSMSS Student Reporter", "student", "Computer Science & Engineering", now_iso)
            ]
            conn.executemany("""
                INSERT INTO users (username, password_hash, salt, full_name, role, department, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, seed_users)

        # Seed authorized sample vehicles
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles;")
        if cursor.fetchone()[0] == 0:
            now_iso = datetime.now(timezone.utc).isoformat()
            sample_vehicles = [
                ("MH20EJ1234", "Dr. S. K. Shinde (Principal)", "Faculty", "Administration", "Car", now_iso),
                ("MH20DE5678", "Prof. R. M. Kulkarni (HOD CSE)", "Faculty", "Computer Science", "Car", now_iso),
                ("MH20BY9012", "CSMSS Campus Ambulance 1", "Emergency", "Hospital Cell", "Ambulance", now_iso),
                ("MH12AB3456", "Estate Maintenance Truck", "Utility", "Campus Maintenance", "Van", now_iso)
            ]
            conn.executemany("""
                INSERT INTO registered_vehicles (plate_number, owner_name, owner_type, department, vehicle_type, registered_at)
                VALUES (?, ?, ?, ?, ?, ?);
            """, sample_vehicles)

# ---------------------------------------------------------------------------
# FastAPI App Initialization & OpenAPI Metadata
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CSMSS Kanchanwadi Civic & Security Monitoring System",
    description=(
        "AI-Powered Civic & Security Monitoring System for CSMSS Chh. Shahu College of Engineering, "
        "Kanchanwadi, Chhatrapati Sambhajinagar. Engineered by Aditya Narendra Bhalerao.\n\n"
        "Features JWT authentication, Role-Based Access Control (RBAC), SQLite WAL concurrency, "
        "and YOLOv8/OpenCV inference hooks."
    ),
    version="2.1.0",
    contact={
        "name": "Aditya Narendra Bhalerao",
        "url": "https://github.com/Aditya-Codebyte-bits/csmss-kanchanwadi-civic-monitor",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    role: str = Field("student", regex="^(admin|security|student|faculty)$")
    department: str = Field("Engineering")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    full_name: str
    maker_credit: str = "Aditya Narendra Bhalerao"

class UserProfileResponse(BaseModel):
    user_id: int
    username: str
    full_name: str
    role: str
    department: str

class IncidentStatusUpdateRequest(BaseModel):
    status: str = Field(..., regex="^(Open|In Progress|Resolved|Closed)$")
    notes: Optional[str] = None

# ---------------------------------------------------------------------------
# Authentication & RBAC Dependencies
# ---------------------------------------------------------------------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_jwt_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject identity")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, full_name, role, department, is_active FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user or not user["is_active"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive or not found")
        return dict(user)

def require_role(allowed_roles: List[str]):
    def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of roles: {allowed_roles}. Current role: '{current_user['role']}'"
            )
        return current_user
    return role_checker

# ---------------------------------------------------------------------------
# File Upload Validation Utility
# ---------------------------------------------------------------------------
def validate_uploaded_image(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Supported image formats: {list(ALLOWED_MIME_TYPES)}"
        )
    file.file.seek(0, os.SEEK_END)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.2f}MB) exceeds maximum limit of {MAX_FILE_SIZE_MB}MB."
        )

# ---------------------------------------------------------------------------
# System Startup Event
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    init_database()

# ---------------------------------------------------------------------------
# Core Public & Attribution Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/credits", tags=["Attribution"])
def get_credits():
    """Returns official creator attribution and academic project credits."""
    return {
        "project": "CSMSS Kanchanwadi Civic & Security Monitoring System",
        "institution": "CSMSS Chh. Shahu College of Engineering, Kanchanwadi, Chhatrapati Sambhajinagar (Aurangabad), Maharashtra",
        "maker": "Aditya Narendra Bhalerao",
        "role": "Lead Architect & Developer",
        "version": "2.1.0",
        "academic_purpose": "Community Involved Project (CIP — 150 Marks)",
        "tech_stack": ["FastAPI", "SQLite WAL", "Streamlit", "YOLOv8", "OpenCV", "EasyOCR"],
        "repository": "https://github.com/Aditya-Codebyte-bits/csmss-kanchanwadi-civic-monitor"
    }

@app.get("/api/v1/health", tags=["Telemetry"])
def health_check():
    return {
        "status": "online",
        "system": "CSMSS Kanchanwadi Civic Monitor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database_engine": "SQLite 3 (WAL mode)",
        "maker_credit": "Aditya Narendra Bhalerao"
    }

# ---------------------------------------------------------------------------
# Auth Endpoints (JWT & RBAC)
# ---------------------------------------------------------------------------
@app.post("/api/v1/auth/token", response_model=TokenResponse, tags=["Authentication"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, password_hash, salt, full_name, role, is_active FROM users WHERE username = ?", (form_data.username,))
        user = cursor.fetchone()
        
        if not user or not verify_password(form_data.password, user["password_hash"], user["salt"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user["is_active"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

        token_payload = {
            "sub": user["username"],
            "user_id": user["user_id"],
            "role": user["role"],
            "full_name": user["full_name"]
        }
        token = create_jwt_token(token_payload)

        return TokenResponse(
            access_token=token,
            role=user["role"],
            username=user["username"],
            full_name=user["full_name"],
            maker_credit="Aditya Narendra Bhalerao"
        )

@app.post("/api/v1/auth/register", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
def register_user(req: UserRegisterRequest):
    pwd_hash, salt = hash_password(req.password)
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO users (username, password_hash, salt, full_name, role, department, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (req.usernam