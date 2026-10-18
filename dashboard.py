"""
Module 3: Dual-Mode Frontend Application (dashboard.py)
Project: CSMSS Chh. Shahu College of Engineering (Kanchanwadi) Civic Monitoring System
Course: Community Involved Project (CIP - 150 Marks)

Description:
Streamlit interactive web application providing two role-based operational views:
  1. Student / Faculty Community Reporter Portal:
     - Upload or capture photos of unlawful parking or campus potholes
     - Instant YOLOv8/OpenCV visual bounding-box feedback & severity grading
     - Live status tracker for submitted campus civic complaints
  2. Campus Security & Estate Office Command Center:
     - Interactive CSMSS Kanchanwadi geospatial heatmap (OpenStreetMap / PyDeck)
     - Real-time Main Gate ANPR license plate scanner & whitelist verifier
     - Admin resolution queue with one-click status updates & CSV audit export

Zero-Cost Resilience:
Operates seamlessly either via the FastAPI REST server (`http://localhost:8000`) OR
directly against the local SQLite database and `CSMSSInferenceEngine` in standalone mode,
so you only need a single terminal command (`streamlit run dashboard.py`) during your viva.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app import (
    DATA_DIR,
    UPLOAD_DIR,
    get_db,
    get_zone_or_default,
    init_db,
    list_campus_zones,
    list_incidents,
    list_registered_vehicles,
    lookup_vehicle,
    ml_engine,
    update_incident_status,
    verify_vehicle_plate,
    IncidentStatusUpdate,
    VehicleCreateRequest,
    register_new_vehicle,
    get_campus_analytics,
)

st.set_page_config(
    page_title="CSMSS Kanchanwadi Civic Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_seeded() -> None:
    """Automatically initializes and seeds the SQLite database if empty."""
    init_db()
    zones = list_campus_zones()
    if not zones:
        try:
            from seed_and_test import seed_database_and_samples

            seed_database_and_samples()
        except Exception as exc:
            st.warning(f"Database initialization note: {exc}")


def run_direct_analysis(
    incident_type: str,
    zone_id: int,
    reported_by_user_id: int,
    image_bytes: bytes,
    filename: str,
    hint_plate: Optional[str] = None,
    user_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes an uploaded image through the ML pipeline and logs it in SQLite."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
    saved_path = UPLOAD_DIR / safe_name
    with open(saved_path, "wb") as f:
        f.write(image_bytes)

    with get_db() as conn:
        zone = get_zone_or_default(conn, zone_id)
        roi_coords = tuple(json.loads(zone["roi_polygon_json"]))
        crit_weight = float(zone["criticality_weight"])

        if incident_type == "unlawful_parking":
            res = ml_engine.detect_unlawful_parking(
                str(saved_path),
                roi_normalized=roi_coords,
                zone_criticality=crit_weight,
                hint_plate=hint_plate,
            )
        elif incident_type == "road_damage":
            res = ml_engine.detect_road_damage(
                str(saved_path),
                zone_criticality=crit_weight,
            )
        else:
            res = ml_engine.extract_license_plate(
                str(saved_path),
                hint_plate=hint_plate,
            )

        vehicle_record = lookup_vehicle(conn, res.detected_plate)
        is_authorized: Optional[bool] = None
        if res.detected_plate:
            is_authorized = bool(vehicle_record is not None)
            if incident_type == "unauthorized_vehicle":
                if not is_authorized:
                    res.severity = "Critical" if crit_weight >= 1.5 else "High"
                    res.summary = (
                        f"ALERT: Unregistered outsider vehicle ({res.detected_plate}) "
                        f"flagged at {zone['zone_name']}."
                    )
                else:
                    res.severity = "Low"
                    res.summary = (
                        f"Authorized {vehicle_record['owner_role']} vehicle ({res.detected_plate}) "
                        f"- {vehicle_record['owner_name']} ({vehicle_record['department']})."
                    )

        desc = f"{res.summary} | Note: {user_notes}" if user_notes else res.summary
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
                desc,
            ),
        )
        incident_id = cur.lastrowid

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
                    res.recommended_action or "Schedule civil inspection.",
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


# -----------------------------------------------------------------------------
# Main UI Layout
# -----------------------------------------------------------------------------
ensure_seeded()

st.sidebar.title("🛡️ CSMSS Civic Monitor")
st.sidebar.caption("CSMSS Chh. Shahu College of Engineering, Kanchanwadi, Chhatrapati Sambhajinagar")

persona_mode = st.sidebar.radio(
    "Select Operating Persona:",
    options=[
        "🎓 Student / Staff Civic Reporter",
        "🚨 Security & Estate Command Center",
        "🚗 Gate ANPR & Vehicle Registry",
    ],
)

zones_list = list_campus_zones()
zone_map = {f"{z['zone_name']} ({z['zone_code']})": z for z in zones_list}
analytics = get_campus_analytics()

st.sidebar.markdown("### Campus Live Pulse")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Open Issues", analytics["active_open_incidents"])
col_s2.metric("Resolved", analytics["resolved_incidents"])
st.sidebar.metric("Outsider Flags", analytics["unauthorized_outsider_flags"])
st.sidebar.info(f"ML Engine: `{ml_engine.engine_mode}` (Zero-Cost CPU Mode)")


# =============================================================================
# VIEW 1: STUDENT / STAFF COMMUNITY REPORTER PORTAL
# =============================================================================
if persona_mode == "🎓 Student / Staff Civic Reporter":
    st.title("🎓 CSMSS Community Civic Reporting Portal")
    st.write(
        "Report **Unlawful Vehicle Parking** or **Internal Road Damage (Potholes/Cracks)** across "
        "the Kanchanwadi campus. Uploaded images are automatically analyzed and geo-tagged for the "
        "Campus Security Officer or Civil Maintenance Cell."
    )

    col_form, col_preview = st.columns([1.1, 1.0], gap="large")

    with col_form:
        st.subheader("1. Submit New Campus Incident")
        selected_task_label = st.selectbox(
            "Issue Category",
            options=[
                "Unlawful Parking (Walkway / Fire Exit / Link Road Obstruction)",
                "Road Surface Damage (Pothole / Monsoon Crack / Waterlogging)",
                "Suspicious / Unregistered Outsider Vehicle (ANPR Audit)",
            ],
        )
        task_map = {
            "Unlawful Parking (Walkway / Fire Exit / Link Road Obstruction)": "unlawful_parking",
            "Road Surface Damage (Pothole / Monsoon Crack / Waterlogging)": "road_damage",
            "Suspicious / Unregistered Outsider Vehicle (ANPR Audit)": "unauthorized_vehicle",
        }
        chosen_task = task_map[selected_task_label]

        selected_zone_label = st.selectbox("CSMSS Kanchanwadi Campus Location", options=list(zone_map.keys()))
        chosen_zone = zone_map[selected_zone_label]

        input_source = st.radio(
            "Image Input Source (Viva Demo Switch):",
            options=["Use Pre-Loaded CSMSS Test Sample", "Upload Photo / Live Camera Capture"],
            horizontal=True,
        )

        sample_dir = DATA_DIR / "samples"
        sample_files = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
        chosen_bytes: Optional[bytes] = None
        chosen_filename: str = "capture.jpg"

        if input_source == "Use Pre-Loaded CSMSS Test Sample" and sample_files:
            sample_names = [p.name for p in sample_files]
            picked_sample = st.selectbox("Select Kanchanwadi Synthetic Test Case:", options=sample_names)
            sample_path = sample_dir / picked_sample
            chosen_bytes = sample_path.read_bytes()
            chosen_filename = picked_sample
        else:
            up_file = st.file_uploader("Upload JPG/PNG frame from phone or webcam", type=["jpg", "jpeg", "png"])
            if up_file is not None:
                chosen_bytes = up_file.getvalue()
                chosen_filename = up_file.name

        optional_plate_hint = st.text_input(
            "Optional Plate Hint (if visible in blurry mobile photo, e.g., MH20EE4512):",
            value="",
        )
        reporter_notes = st.text_area(
            "Additional Landmark Details:",
            placeholder="e.g., Blocking ambulance bay near Ayurveda Hospital link gate...",
        )

        submit_btn = st.button("🔍 Run AI Detection & Log Incident", type="primary", use_container_width=True)

    with col_preview:
        st.subheader("2. Automated AI Inspection Output")
        if submit_btn and chosen_bytes:
            with st.spinner("Running YOLOv8 / OpenCV Computer Vision pipeline..."):
                outcome = run_direct_analysis(
                    incident_type=chosen_task,
                    zone_id=chosen_zone["zone_id"],
                    reported_by_user_id=1,
                    image_bytes=chosen_bytes,
                    filename=chosen_filename,
                    hint_plate=optional_plate_hint.strip() or None,
                    user_notes=reporter_notes.strip() or None,
                )
                inf = outcome["inference"]
                st.success(f"Incident #{outcome['incident_id']} logged at **{chosen_zone['zone_name']}**!")

                m1, m2, m3 = st.columns(3)
                m1.metric("Severity Level", inf["severity"])
                m2.metric("AI Confidence", f"{int(inf['confidence'] * 100)}%")
                m3.metric("Plate Extracted", inf["detected_plate"] or "N/A")

                if os.path.exists(inf["annotated_image_path"]):
                    st.image(
                        inf["annotated_image_path"],
                        caption=f"Annotated Output: {inf['summary']}",
                        use_container_width=True,
                    )

                if outcome["registered_vehicle_info"]:
                    rv = outcome["registered_vehicle_info"]
                    st.info(
                        f"**Registered CSMSS Owner Identified:** {rv['owner_name']} "
                        f"({rv['owner_role'].upper()} - {rv['department']}) | Pass: `{rv['pass_sticker_id']}`"
                    )
                elif inf["detected_plate"]:
                    st.error(
                        f"**UNREGISTERED OUTSIDER VEHICLE:** Plate `{inf['detected_plate']}` is NOT "
                        f"in the CSMSS authorized vehicle registry!"
                    )

                if inf.get("recommended_action"):
                    st.warning(f"**Recommended Action:** {inf['recommended_action']}")
        elif chosen_bytes:
            st.image(chosen_bytes, caption="Selected Input Frame (Ready for Analysis)", use_container_width=True)
        else:
            st.info("Select a CSMSS test image or upload a photo on the left to run real-time detection.")


# =============================================================================
# VIEW 2: SECURITY & ESTATE OFFICE COMMAND CENTER
# =============================================================================
elif persona_mode == "🚨 Security & Estate Command Center":
    st.title("🚨 CSMSS Security & Civil Estate Command Center")
    st.write(
        "Real-time geospatial incident heatmap across **CSMSS Kanchanwadi Campus** (`19.8162° N, 75.2914° E`), "
        "prioritized hazard queue, and administrative resolution workflow."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Logged Incidents", analytics["total_incidents"])
    k2.metric("Pending Action (Open)", analytics["active_open_incidents"])
    k3.metric("Resolved & Verified", analytics["resolved_incidents"])
    k4.metric("Outsider Vehicle Alerts", analytics["unauthorized_outsider_flags"])

    all_incidents = list_incidents()

    col_map, col_queue = st.columns([1.15, 1.0], gap="large")

    with col_map:
        st.subheader("📍 CSMSS Kanchanwadi Geospatial Incident Map")
        if all_incidents:
            map_df = pd.DataFrame(all_incidents)[
                ["latitude", "longitude", "zone_name", "incident_type", "severity", "status"]
            ]
            st.map(map_df, latitude="latitude", longitude="longitude", size=35)
        else:
            st.info("No geo-tagged incidents available yet.")

        st.subheader("📊 Incident Breakdown by Campus Zone")
        zone_summary_df = pd.DataFrame(analytics["zone_heatmap_summary"])
        st.dataframe(zone_summary_df, use_container_width=True, hide_index=True)

    with col_queue:
        st.subheader("🛠️ Admin Resolution & Dispatch Workflow")
        filter_status = st.selectbox("Filter Queue by Status:", ["All", "Open", "In_Progress", "Resolved", "Dismissed"])
        filtered_inc = [
            inc for inc in all_incidents if filter_status == "All" or inc["status"] == filter_status
        ]

        if filtered_inc:
            inc_options = {
                f"#{inc['incident_id']} [{inc['severity']}] {inc['incident_type']} @ {inc['zone_name']} ({inc['status']})": inc
                for inc in filtered_inc
            }
            selected_inc_label = st.selectbox("Select Incident to Inspect / Resolve:", list(inc_options.keys()))
            active_inc = inc_options[selected_inc_label]

            st.markdown(f"**Summary:** {active_inc['description']}")
            st.caption(
                f"Reported by: {active_inc['reporter_name']} ({active_inc['reporter_role']}) | "
                f"Confidence: {int(active_inc['confidence_score'] * 100)}% | Created: {active_inc['created_at']}"
            )

            if active_inc.get("estimated_diameter_cm"):
                st.warning(
                    f"**Civil Defect Metrics:** {active_inc['defect_category']} | "
                    f"Est. Span: `{active_inc['estimated_diameter_cm']} cm` | "
                    f"Action: {active_inc['recommended_action']}"
                )

            if os.path.exists(active_inc["image_path"]):
                st.image(active_inc["image_path"], width=460)

            new_status = st.selectbox(
                "Update Workflow Status:",
                options=["Open", "In_Progress", "Resolved", "Dismissed"],
                index=["Open", "In_Progress", "Resolved", "Dismissed"].index(active_inc["status"]),
            )
            res_notes = st.text_input(
                "Official Resolution / Security Action Notes:",
                value=active_inc.get("resolution_notes") or "Cleared obstruction / Work order issued to Civil Cell.",
            )
            if st.button("💾 Save Resolution Update", type="primary"):
                update_incident_status(
                    active_inc["incident_id"],
                    IncidentStatusUpdate(status=new_status, resolution_notes=res_notes),
                )
                st.success(f"Incident #{active_inc['incident_id']} updated to **{new_status}**!")
                st.rerun()

    st.subheader("📑 Master Geo-Tagged Audit Log (Exportable for CIP Report)")
    if all_incidents:
        export_df = pd.DataFrame(all_incidents)
        st.dataframe(
            export_df[
                [
                    "incident_id",
                    "incident_type",
                    "zone_name",
                    "severity",
                    "status",
                    "detected_plate",
                    "is_authorized_vehicle",
                    "confidence_score",
                    "description",
                    "created_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSMSS Audit Log (CSV)",
            data=csv_bytes,
            file_name="csmss_kanchanwadi_civic_audit.csv",
            mime="text/csv",
        )


# =============================================================================
# VIEW 3: GATE ANPR & AUTHORIZED VEHICLE REGISTRY
# =============================================================================
else:
    st.title("🚗 CSMSS Main Gate ANPR & Vehicle Whitelist Registry")
    st.write(
        "Cross-verify entering vehicles against the authorized **Student, Faculty, College Bus, and "
        "Ayurveda Hospital Ambulance** database."
    )

    col_verify, col_reg = st.columns([1.0, 1.2], gap="large")

    with col_verify:
        st.subheader("🔎 Instant Plate Lookup (Gate Security Desk)")
        query_plate = st.text_input("Enter or Scan License Plate Number:", value="MH20EE4512")
        if st.button("Verify Plate Authorization", type="primary"):
            result = verify_vehicle_plate(query_plate)
            if result["is_authorized"]:
                v = result["vehicle_details"]
                st.success(f"✅ AUTHORIZED: `{result['plate_number']}`")
                st.json(v)
            else:
                st.error(
                    f"🚨 UNREGISTERED OUTSIDER: `{result['plate_number']}` is NOT in the CSMSS registry! "
                    f"Log visitor entry at Main Gate."
                )

        st.subheader("➕ Register New Campus Vehicle Pass")
        with st.form("new_vehicle_form"):
            p_num = st.text_input("License Plate (e.g., MH20GH7788)")
            o_name = st.text_input("Student / Faculty Full Name")
            o_role = st.selectbox("Owner Role", ["student", "faculty", "staff", "college_bus", "ambulance"])
            dept = st.text_input("Department (e.g., CSE, AI&DS, Mechanical, Civil)")
            v_type = st.selectbox("Vehicle Type", ["two_wheeler", "four_wheeler", "bus", "emergency"])
            sticker = st.text_input("CSMSS Pass Sticker ID (e.g., CSMSS-PASS-2026-501)")
            phone = st.text_input("Contact Phone Number")
            reg_submit = st.form_submit_button("Register Vehicle in Whitelist")

            if reg_submit and p_num and o_name and sticker:
                try:
                    register_new_vehicle(
                        VehicleCreateRequest(
                            plate_number=p_num,
                            owner_name=o_name,
                            owner_role=o_role,
                            department=dept or "Engineering",
                            vehicle_type=v_type,
                            pass_sticker_id=sticker,
                            contact_phone=phone,
                        )
                    )
                    st.success(f"Registered `{p_num}` with pass `{sticker}`!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Registration error: {exc}")

    with col_reg:
        st.subheader("📋 Authorized CSMSS Vehicle Registry")
        vehicles = list_registered_vehicles()
        if vehicles:
            st.dataframe(pd.DataFrame(vehicles), use_container_width=True, hide_index=True)
