"""
Module 4: Seed Data & Automated End-to-End Testing Script (seed_and_test.py)
Project: CSMSS Chh. Shahu College of Engineering (Kanchanwadi) Civic Monitoring System
Course: Community Involved Project (CIP - 150 Marks)

Description:
1. Populates `csmss_civic.db` with realistic Kanchanwadi campus zones, users, and
   Chhatrapati Sambhajinagar (`MH20`) student/faculty/bus/ambulance vehicle records.
2. Generates 4 high-contrast synthetic test images (`.jpg`) using OpenCV so you have
   ready-to-demo visual scenarios right out of the box:
   - Scenario 1: Unlawful car parking inside the Engineering Workshop Fire Exit ROI
   - Scenario 2: Severe monsoon pothole on the Ayurveda Hospital–Engineering Link Road
   - Scenario 3: Authorized Faculty vehicle (`MH20EE4512`) entering Main Gate
   - Scenario 4: Unregistered Outsider vehicle (`MH12XY9999`) flagged at Main Gate
3. Runs an automated verification suite across both the ML Inference Engine and the
   FastAPI REST endpoints (`TestClient`) to guarantee everything works before your viva.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import DATA_DIR, app, get_db, init_db

SAMPLE_DIR = DATA_DIR / "samples"


def generate_synthetic_campus_images() -> Dict[str, Path]:
    """Creates 4 realistic synthetic test images for offline viva demonstration."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. Unlawful Parking inside Restricted Fire Exit Zone (with plate MH20BY1122)
    img_park = np.full((480, 640, 3), (210, 215, 215), dtype=np.uint8)
    # Road asphalt background
    cv2.rectangle(img_park, (40, 80), (600, 440), (95, 95, 95), -1)
    # Draw distinct vehicle silhouette inside the central ROI
    cv2.rectangle(img_park, (180, 150), (460, 360), (180, 50, 30), -1)
    cv2.rectangle(img_park, (205, 175), (435, 335), (25, 25, 25), 3)
    # High-contrast license plate rectangle on vehicle bumper
    cv2.rectangle(img_park, (245, 300), (395, 342), (250, 250, 250), -1)
    cv2.rectangle(img_park, (245, 300), (395, 342), (10, 10, 10), 2)
    cv2.putText(img_park, "MH20BY1122", (252, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    park_path = SAMPLE_DIR / "01_unlawful_parking_MH20BY1122.jpg"
    cv2.imwrite(str(park_path), img_park)
    paths["unlawful_parking"] = park_path

    # 2. Severe Road Pothole on Hospital-Engineering Link Road
    img_pothole = np.full((480, 640, 3), (145, 145, 145), dtype=np.uint8)
    # Simulate textured gray asphalt surface
    noise = np.random.randint(-12, 12, (480, 640, 3), dtype=np.int16)
    img_pothole = np.clip(img_pothole.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Draw dark irregular pothole cavity in the road center
    cv2.ellipse(img_pothole, (320, 280), (115, 65), 12, 0, 360, (22, 22, 22), -1)
    cv2.ellipse(img_pothole, (310, 275), (95, 50), 12, 0, 360, (12, 12, 12), -1)
    pothole_path = SAMPLE_DIR / "02_road_pothole_hospital_link.jpg"
    cv2.imwrite(str(pothole_path), img_pothole)
    paths["road_damage"] = pothole_path

    # 3. Authorized Faculty Vehicle at Main Gate (MH20EE4512)
    img_auth = np.full((480, 640, 3), (190, 200, 195), dtype=np.uint8)
    cv2.rectangle(img_auth, (140, 120), (500, 390), (60, 90, 140), -1)
    cv2.rectangle(img_auth, (210, 270), (430, 335), (255, 255, 255), -1)
    cv2.rectangle(img_auth, (210, 270), (430, 335), (0, 0, 0), 3)
    cv2.putText(img_auth, "MH20EE4512", (222, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3)
    auth_path = SAMPLE_DIR / "03_gate_authorized_MH20EE4512.jpg"
    cv2.imwrite(str(auth_path), img_auth)
    paths["authorized_gate"] = auth_path

    # 4. Unregistered Outsider Vehicle at Main Gate (MH12XY9999)
    img_outsider = np.full((480, 640, 3), (185, 185, 195), dtype=np.uint8)
    cv2.rectangle(img_outsider, (140, 120), (500, 390), (90, 50, 50), -1)
    cv2.rectangle(img_outsider, (210, 270), (430, 335), (255, 255, 255), -1)
    cv2.rectangle(img_outsider, (210, 270), (430, 335), (0, 0, 0), 3)
    cv2.putText(img_outsider, "MH12XY9999", (222, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3)
    outsider_path = SAMPLE_DIR / "04_gate_outsider_MH12XY9999.jpg"
    cv2.imwrite(str(outsider_path), img_outsider)
    paths["outsider_gate"] = outsider_path

    return paths


def seed_database_and_samples() -> Dict[str, Path]:
    """Seeds users, CSMSS Kanchanwadi zones, vehicle whitelist, and initial incidents."""
    init_db()
    sample_paths = generate_synthetic_campus_images()

    users_seed = [
        ("Aarav Kulkarni", "student", "CSMSS-CSE-2026-014", "9823011223"),
        ("Prof. S. B. Deshmukh", "faculty", "CSMSS-FAC-CSE-008", "9422200112"),
        ("Rameshwar Jadhav (Main Gate Shift A)", "security", "CSMSS-SEC-001", "9765400987"),
        ("Er. V. K. Patil (Estate & Civil Cell)", "civil_cell", "CSMSS-EST-002", "9850011445"),
        ("Dr. Admin Coordinator", "admin", "CSMSS-ADM-001", "9422711220"),
    ]

    # Real GPS coordinates around CSMSS Chh. Shahu College of Engineering, Kanchanwadi
    zones_seed = [
        (
            "ZONE-GATE-01",
            "CSMSS Main Entrance Gate (Paithan Link Road)",
            19.8168,
            75.2912,
            1,
            1.6,
            json.dumps([0.15, 0.20, 0.85, 0.88]),
        ),
        (
            "ZONE-ENG-FIRE",
            "Engineering Main Block & Workshop Fire Exit",
            19.8161,
            75.2918,
            1,
            1.8,
            json.dumps([0.12, 0.18, 0.88, 0.86]),
        ),
        (
            "ZONE-HOSP-LINK",
            "Ayurveda Hospital to Engineering Internal Link Road",
            19.8156,
            75.2909,
            1,
            2.0,
            json.dumps([0.10, 0.15, 0.90, 0.90]),
        ),
        (
            "ZONE-FAC-BAY",
            "Reserved Faculty & HOD Parking Bay",
            19.8164,
            75.2923,
            1,
            1.3,
            json.dumps([0.15, 0.22, 0.85, 0.85]),
        ),
        (
            "ZONE-HOSTEL-RD",
            "Polytechnic & Student Hostel Approach Road",
            19.8152,
            75.2920,
            0,
            1.2,
            json.dumps([0.15, 0.20, 0.85, 0.85]),
        ),
    ]

    vehicles_seed = [
        ("MH20EE4512", "Prof. S. B. Deshmukh", "faculty", "CSE", "four_wheeler", "CSMSS-PASS-2026-001", "9422200112"),
        ("MH20BY1122", "Aarav Kulkarni", "student", "CSE", "two_wheeler", "CSMSS-PASS-2026-104", "9823011223"),
        ("MH20DV8801", "Snehal Chavan", "student", "AI&DS", "two_wheeler", "CSMSS-PASS-2026-118", "9763312345"),
        ("MH20AZ9090", "CSMSS Campus Bus #4", "college_bus", "Transport Cell", "bus", "CSMSS-BUS-004", "9422090900"),
        ("MH20AM1080", "CSMSS Rugnalaya Emergency Van", "ambulance", "Ayurveda Hospital", "emergency", "CSMSS-EMG-108", "9422010800"),
        ("MH20CT3344", "Prof. M. R. Shinde", "faculty", "Mechanical", "four_wheeler", "CSMSS-PASS-2026-019", "9890122334"),
    ]

    with get_db() as conn:
        for u in users_seed:
            conn.execute(
                """
                INSERT OR IGNORE INTO users (full_name, role, department_or_id, contact_phone)
                VALUES (?, ?, ?, ?)
                """,
                u,
            )

        for z in zones_seed:
            conn.execute(
                """
                INSERT OR IGNORE INTO campus_zones
                (zone_code, zone_name, latitude, longitude, is_no_parking_zone, criticality_weight, roi_polygon_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                z,
            )

        for v in vehicles_seed:
            conn.execute(
                """
                INSERT OR IGNORE INTO registered_vehicles
                (plate_number, owner_name, owner_role, department, vehicle_type, pass_sticker_id, contact_phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                v,
            )
        conn.commit()

    return sample_paths


def run_end_to_end_verification() -> None:
    """Executes automated tests across all 3 functional modules via FastAPI TestClient."""
    print("=" * 78)
    print("CSMSS CIVIC MONITOR - SEEDING & END-TO-END VERIFICATION SUITE")
    print("=" * 78)

    sample_paths = seed_database_and_samples()
    client = TestClient(app)

    # 1. Verify Health & Zones
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
    print(f"[PASS] 1. Health Endpoint: {r_health.json()['status']} | Engine: {r_health.json()['ml_engine_mode']}")

    r_zones = client.get("/api/v1/zones")
    assert r_zones.status_code == 200 and len(r_zones.json()) >= 5
    print(f"[PASS] 2. Campus Zones Seeded: {len(r_zones.json())} Kanchanwadi zones active.")

    # 2. Test Module 1: Unlawful Parking Detection at Engineering Fire Exit (Zone 2)
    with open(sample_paths["unlawful_parking"], "rb") as f:
        r_park = client.post(
            "/api/v1/analyze/unlawful_parking",
            data={
                "zone_id": 2,
                "reported_by_user_id": 1,
                "hint_plate": "MH20BY1122",
                "user_notes": "Student two-wheeler/car blocking Workshop Fire Exit.",
            },
            files={"file": (sample_paths["unlawful_parking"].name, f, "image/jpeg")},
        )
    assert r_park.status_code == 200, r_park.text
    park_data = r_park.json()
    assert park_data["inference"]["detected"] is True
    print(
        f"[PASS] 3. Unlawful Parking Pipeline: Incident #{park_data['incident_id']} | "
        f"Severity={park_data['inference']['severity']} | Owner={park_data['registered_vehicle_info']['owner_name']}"
    )

    # 3. Test Module 2: Road Damage (Pothole) Detection on Hospital Link Road (Zone 3)
    with open(sample_paths["road_damage"], "rb") as f:
        r_road = client.post(
            "/api/v1/analyze/road_damage",
            data={
                "zone_id": 3,
                "reported_by_user_id": 2,
                "user_notes": "Deep monsoon pothole slowing ambulance movement.",
            },
            files={"file": (sample_paths["road_damage"].name, f, "image/jpeg")},
        )
    assert r_road.status_code == 200, r_road.text
    road_data = r_road.json()
    assert road_data["inference"]["detected"] is True
    print(
        f"[PASS] 4. Road Damage Pipeline: Incident #{road_data['incident_id']} | "
        f"Defect={road_data['inference']['defect_category']} | "
        f"Span=~{road_data['inference']['estimated_diameter_cm']} cm | "
        f"Severity={road_data['inference']['severity']}"
    )

    # 4. Test Module 3A: Authorized Faculty Vehicle ANPR at Main Gate (Zone 1)
    with open(sample_paths["authorized_gate"], "rb") as f:
        r_auth = client.post(
            "/api/v1/analyze/unauthorized_vehicle",
            data={"zone_id": 1, "reported_by_user_id": 3, "hint_plate": "MH20EE4512"},
            files={"file": (sample_paths["authorized_gate"].name, f, "image/jpeg")},
        )
    assert r_auth.status_code == 200, r_auth.text
    auth_data = r_auth.json()
    assert auth_data["is_authorized_vehicle"] is True
    print(
        f"[PASS] 5. Gate ANPR (Authorized): Plate={auth_data['inference']['detected_plate']} -> "
        f"Verified {auth_data['registered_vehicle_info']['owner_name']} ({auth_data['registered_vehicle_info']['department']})"
    )

    # 5. Test Module 3B: Unregistered Outsider Vehicle ANPR at Main Gate (Zone 1)
    with open(sample_paths["outsider_gate"], "rb") as f:
        r_out = client.post(
            "/api/v1/analyze/unauthorized_vehicle",
            data={"zone_id": 1, "reported_by_user_id": 3, "hint_plate": "MH12XY9999"},
            files={"file": (sample_paths["outsider_gate"].name, f, "image/jpeg")},
        )
    assert r_out.status_code == 200, r_out.text
    out_data = r_out.json()
    assert out_data["is_authorized_vehicle"] is False
    print(
        f"[PASS] 6. Gate ANPR (Outsider Alert): Plate={out_data['inference']['detected_plate']} -> "
        f"Flagged as UNREGISTERED OUTSIDER (Severity={out_data['inference']['severity']})"
    )

    # 6. Test Admin Resolution Workflow
    r_resolve = client.patch(
        f"/api/v1/incidents/{park_data['incident_id']}/resolve",
        json={
            "status": "Resolved",
            "resolution_notes": "Security guard contacted student owner (Aarav Kulkarni); vehicle moved within 8 mins.",
        },
    )
    assert r_resolve.status_code == 200 and r_resolve.json()["status"] == "Resolved"
    print(f"[PASS] 7. Admin Resolution Workflow: Incident #{park_data['incident_id']} marked Resolved.")

    # 7. Verify Analytics Summary
    r_stats = client.get("/api/v1/analytics/summary")
    assert r_stats.status_code == 200
    stats = r_stats.json()
    print(
        f"[PASS] 8. Analytics Summary: Total={stats['total_incidents']}, "
        f"Open={stats['active_open_incidents']}, Resolved={stats['resolved_incidents']}, "
        f"Outsiders={stats['unauthorized_outsider_flags']}"
    )
    print("=" * 78)
    print("ALL 8 END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_end_to_end_verification()
