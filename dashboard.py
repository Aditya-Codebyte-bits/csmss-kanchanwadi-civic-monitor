"""
Module 3: Minimal Technical Frontend Console (dashboard.py)
Project: CSMSS Chh. Shahu College of Engineering (Kanchanwadi) Civic Monitoring System
Course: Community Involved Project (CIP - 150 Marks)

Description:
Engineered with a minimal, technical telemetry aesthetic (Linear / Palantir style):
  - Full-width layout (`initial_sidebar_state="collapsed"`) so mobile & laptop screens
    never suffer from cramped multi-column text wrapping.
  - Monospace telemetry bar, bordered KPI cards, clean top horizontal tabs, and
    calibrated geospatial markers (`size=12`).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from app import (
    DATA_DIR,
    UPLOAD_DIR,
    IncidentStatusUpdate,
    VehicleCreateRequest,
    get_campus_analytics,
    get_db,
    get_zone_or_default,
    init_db,
    list_campus_zones,
    list_incidents,
    list_registered_vehicles,
    lookup_vehicle,
    ml_engine,
    register_new_vehicle,
    update_incident_status,
    verify_vehicle_plate,
)

st.set_page_config(
    page_title="CSMSS // Civic Telemetry Console",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Minimal Technical CSS Design System
# -----------------------------------------------------------------------------
TECHNICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Remove excessive default top padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1280px !important;
}

/* Technical top Masthead */
.tech-header {
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.03) 0%, rgba(15, 23, 42, 0.08) 100%);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
}

.tech-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 4px;
}

.tech-title {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.3;
}

.tech-subtitle {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
}

/* Minimal KPI Telemetry Strip */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
    margin-bottom: 18px;
}

.kpi-card {
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 6px;
    padding: 10px 14px;
    background: rgba(248, 250, 252, 0.5);
}

.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
}

.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    margin-top: 2px;
}

/* Section Headers */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #334155;
    border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    padding-bottom: 6px;
    margin-bottom: 12px;
    margin-top: 8px;
}

/* Status Pills */
.status-pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid rgba(148, 163, 184, 0.35);
}

/* Override loud default coral buttons with sleek slate-navy */
div.stButton > button[kind="primary"] {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1rem !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #1e293b !important;
    border-color: #334155 !important;
}

/* Clean Tabs */
button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
</style>
"""
st.markdown(TECHNICAL_CSS, unsafe_allow_html=True)


def ensure_seeded() -> None:
    """Automatically initializes and seeds the SQLite database & baseline incidents if empty."""
    init_db()
    zones = list_campus_zones()
    incidents = list_incidents()
    if not zones or not incidents:
        try:
            from seed_and_test import seed_database_and_samples

            seed_database_and_samples(seed_initial_incidents=True)
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
                        f"UNREGISTERED_OUTSIDER ({res.detected_plate}) "
                        f"flagged at {zone['zone_code']}."
                    )
                else:
                    res.severity = "Low"
                    res.summary = (
                        f"AUTHORIZED ({res.detected_plate}) // "
                        f"{vehicle_record['owner_name']} [{vehicle_record['department']}]."
                    )

        desc = f"{res.summary} | NOTE: {user_notes}" if user_notes else res.summary
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
# Initialize State & Top Telemetry Masthead
# -----------------------------------------------------------------------------
ensure_seeded()
zones_list = list_campus_zones()
zone_map = {f"{z['zone_code']} — {z['zone_name']}": z for z in zones_list}
analytics = get_campus_analytics()

st.markdown(
    f"""
    <div class="tech-header">
        <div class="tech-eyebrow">
            SYS_STATUS: ONLINE &nbsp;|&nbsp; ENGINE: {ml_engine.engine_mode.upper()} &nbsp;|&nbsp; LOC: 19.8162°N 75.2914°E
        </div>
        <div class="tech-title">CSMSS KANCHANWADI // CIVIC &amp; ANPR TELEMETRY CONSOLE</div>
        <div class="tech-subtitle">
            Chh. Shahu College of Engineering • Automated Unlawful Parking, Road Surface Defect &amp; Gate Access Auditing
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">TOTAL LOGGED EVENTS</div>
            <div class="kpi-val">{analytics['total_incidents']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ACTIVE / OPEN QUEUE</div>
            <div class="kpi-val">{analytics['active_open_incidents']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">RESOLVED &amp; VERIFIED</div>
            <div class="kpi-val">{analytics['resolved_incidents']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">OUTSIDER ANPR FLAGS</div>
            <div class="kpi-val">{analytics['unauthorized_outsider_flags']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Use clean horizontal tabs instead of a wide sidebar so mobile screens have 100% width
tab_infer, tab_command, tab_anpr = st.tabs(
    [
        "01 // CV INFERENCE & REPORTING",
        "02 // GEOSPATIAL OPS & TRIAGE",
        "03 // ANPR REGISTRY & WHITELIST",
    ]
)

# =============================================================================
# TAB 1: CV INFERENCE & INCIDENT LOGGING
# =============================================================================
with tab_infer:
    col_form, col_preview = st.columns([1.0, 1.15], gap="medium")

    with col_form:
        st.markdown('<div class="sec-label">01.A // INPUT PARAMETERS &amp; TELEMETRY SOURCE</div>', unsafe_allow_html=True)

        selected_task_label = st.selectbox(
            "Detection Pipeline",
            options=[
                "UNLAWFUL_PARKING // Restricted ROI Intersection",
                "ROAD_DAMAGE // Surface Cavity & Crack Analysis",
                "UNAUTHORIZED_VEHICLE // Gate ANPR & Whitelist Audit",
            ],
        )
        task_map = {
            "UNLAWFUL_PARKING // Restricted ROI Intersection": "unlawful_parking",
            "ROAD_DAMAGE // Surface Cavity & Crack Analysis": "road_damage",
            "UNAUTHORIZED_VEHICLE // Gate ANPR & Whitelist Audit": "unauthorized_vehicle",
        }
        chosen_task = task_map[selected_task_label]

        selected_zone_label = st.selectbox("Campus Sector / Zone", options=list(zone_map.keys()))
        chosen_zone = zone_map[selected_zone_label]

        input_source = st.radio(
            "Frame Source",
            options=["Pre-Loaded Kanchanwadi CCTV Frame", "Custom Image Upload"],
            horizontal=True,
        )

        sample_dir = DATA_DIR / "samples"
        sample_files = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
        chosen_bytes: Optional[bytes] = None
        chosen_filename: str = "capture.jpg"

        if input_source == "Pre-Loaded Kanchanwadi CCTV Frame" and sample_files:
            sample_names = [p.name for p in sample_files]
            picked_sample = st.selectbox("Select Test Frame", options=sample_names)
            sample_path = sample_dir / picked_sample
            chosen_bytes = sample_path.read_bytes()
            chosen_filename = picked_sample
        else:
            up_file = st.file_uploader("Upload Frame (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if up_file is not None:
                chosen_bytes = up_file.getvalue()
                chosen_filename = up_file.name

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            optional_plate_hint = st.text_input(
                "Manual OCR Override (Optional)",
                value="",
                placeholder="e.g. MH20EE4512",
            )
        with c_p2:
            reporter_notes = st.text_input(
                "Field Operator Log Note",
                value="",
                placeholder="e.g. Blocking emergency lane",
            )

        submit_btn = st.button("EXECUTE CV PIPELINE & COMMIT LOG", type="primary", use_container_width=True)

    with col_preview:
        st.markdown('<div class="sec-label">01.B // ANNOTATED COMPUTER VISION OUTPUT</div>', unsafe_allow_html=True)

        if submit_btn and chosen_bytes:
            with st.spinner("Executing inference pipeline..."):
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

                if os.path.exists(inf["annotated_image_path"]):
                    st.image(inf["annotated_image_path"], use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("SEVERITY", inf["severity"].upper())
                m2.metric("CONFIDENCE", f"{inf['confidence']:.2f}")
                m3.metric("PLATE_OCR", inf["detected_plate"] or "NONE")

                st.code(
                    f"INCIDENT_ID: #{outcome['incident_id']}  |  SECTOR: {chosen_zone['zone_code']}\n"
                    f"SUMMARY:     {inf['summary']}\n"
                    f"ACTION:      {inf.get('recommended_action', 'N/A')}",
                    language="yaml",
                )

                if outcome["registered_vehicle_info"]:
                    rv = outcome["registered_vehicle_info"]
                    st.caption(
                        f"WHITELIST_MATCH // OWNER: {rv['owner_name']} ({rv['owner_role'].upper()} - {rv['department']}) "
                        f"| PASS: {rv['pass_sticker_id']} | TEL: {rv['contact_phone']}"
                    )
                elif inf["detected_plate"]:
                    st.caption(
                        f"SECURITY_ALERT // PLATE {inf['detected_plate']} NOT FOUND IN AUTHORIZED REGISTRY"
                    )
        elif chosen_bytes:
            st.image(chosen_bytes, use_container_width=True)
            st.caption(f"READY // SOURCE_FRAME: {chosen_filename} | SECTOR_WEIGHT: {chosen_zone['criticality_weight']}x")
        else:
            st.caption("AWAITING_INPUT // Select a pre-loaded CCTV frame or upload an image.")


# =============================================================================
# TAB 2: GEOSPATIAL COMMAND & TRIAGE
# =============================================================================
with tab_command:
    all_incidents = list_incidents()
    col_map, col_queue = st.columns([1.05, 1.0], gap="medium")

    with col_map:
        st.markdown('<div class="sec-label">02.A // KANCHANWADI SECTOR GEOSPATIAL TELEMETRY</div>', unsafe_allow_html=True)
        if all_incidents:
            map_df = pd.DataFrame(all_incidents)[
                ["latitude", "longitude", "zone_name", "incident_type", "severity", "status"]
            ].copy()
            # Calibrated compact point radius (14m) so markers remain sharp on mobile & desktop
            st.map(map_df, latitude="latitude", longitude="longitude", size=14, zoom=16)
        else:
            st.caption("No geo-tagged incidents logged.")

        st.markdown('<div class="sec-label">02.B // SECTOR DENSITY MATRIX</div>', unsafe_allow_html=True)
        zone_summary_df = pd.DataFrame(analytics["zone_heatmap_summary"])
        st.dataframe(
            zone_summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "zone_name": "Campus Sector",
                "latitude": st.column_config.NumberColumn("Lat", format="%.4f"),
                "longitude": st.column_config.NumberColumn("Lon", format="%.4f"),
                "incident_count": "Events",
            },
        )

    with col_queue:
        st.markdown('<div class="sec-label">02.C // INCIDENT TRIAGE &amp; RESOLUTION DISPATCH</div>', unsafe_allow_html=True)
        filter_status = st.selectbox("Filter Queue by Status", ["All", "Open", "In_Progress", "Resolved", "Dismissed"])
        filtered_inc = [
            inc for inc in all_incidents if filter_status == "All" or inc["status"] == filter_status
        ]

        if filtered_inc:
            inc_options = {
                f"ID #{inc['incident_id']}  |  {inc['severity'].upper()}  |  {inc['incident_type'].upper()}  |  {inc['zone_code']}": inc
                for inc in filtered_inc
            }
            selected_inc_label = st.selectbox("Select Event Record", list(inc_options.keys()))
            active_inc = inc_options[selected_inc_label]

            if os.path.exists(active_inc["image_path"]):
                st.image(active_inc["image_path"], use_container_width=True)

            st.code(
                f"EVENT_ID:   #{active_inc['incident_id']} [{active_inc['status'].upper()}]\n"
                f"TIMESTAMP:  {active_inc['created_at']}\n"
                f"SECTOR:     {active_inc['zone_code']} ({active_inc['zone_name']})\n"
                f"CONFIDENCE: {active_inc['confidence_score']:.2f}  |  PLATE: {active_inc['detected_plate'] or 'N/A'}\n"
                f"DETAIL:     {active_inc['description']}",
                language="yaml",
            )

            if active_inc.get("estimated_diameter_cm"):
                st.caption(
                    f"CIVIL_SPEC // DEFECT: {active_inc['defect_category']} | "
                    f"EST_SPAN: {active_inc['estimated_diameter_cm']} cm | "
                    f"RECOMMENDED: {active_inc['recommended_action']}"
                )

            c_u1, c_u2 = st.columns([1, 2])
            with c_u1:
                new_status = st.selectbox(
                    "Set Status",
                    options=["Open", "In_Progress", "Resolved", "Dismissed"],
                    index=["Open", "In_Progress", "Resolved", "Dismissed"].index(active_inc["status"]),
                )
            with c_u2:
                res_notes = st.text_input(
                    "Resolution Log Note",
                    value=active_inc.get("resolution_notes") or "Obstruction cleared / Civil work order dispatched.",
                )

            if st.button("COMMIT STATUS UPDATE", type="primary", use_container_width=True):
                update_incident_status(
                    active_inc["incident_id"],
                    IncidentStatusUpdate(status=new_status, resolution_notes=res_notes),
                )
                st.rerun()

    st.markdown('<div class="sec-label">02.D // MASTER GEO-TAGGED AUDIT LOG</div>', unsafe_allow_html=True)
    if all_incidents:
        export_df = pd.DataFrame(all_incidents)
        st.dataframe(
            export_df[
                [
                    "incident_id",
                    "created_at",
                    "zone_code",
                    "incident_type",
                    "severity",
                    "status",
                    "detected_plate",
                    "confidence_score",
                    "description",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "incident_id": "ID",
                "created_at": "Timestamp",
                "zone_code": "Sector",
                "incident_type": "Pipeline",
                "severity": "Severity",
                "status": "Status",
                "detected_plate": "Plate",
                "confidence_score": st.column_config.NumberColumn("Conf", format="%.2f"),
                "description": "Telemetry Summary",
            },
        )
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "EXPORT AUDIT LOG (.CSV)",
            data=csv_bytes,
            file_name="csmss_kanchanwadi_civic_audit.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 3: ANPR GATE REGISTRY & WHITELIST
# =============================================================================
with tab_anpr:
    col_verify, col_reg = st.columns([1.0, 1.35], gap="medium")

    with col_verify:
        st.markdown('<div class="sec-label">03.A // INSTANT RTO PLATE VERIFIER</div>', unsafe_allow_html=True)
        query_plate = st.text_input("Query Registration Number", value="MH20EE4512")
        if st.button("VERIFY AUTHORIZATION", type="primary", use_container_width=True):
            result = verify_vehicle_plate(query_plate)
            if result["is_authorized"]:
                v = result["vehicle_details"]
                st.code(
                    f"STATUS:      AUTHORIZED_CSMSS_MEMBER\n"
                    f"PLATE_NO:    {result['plate_number']}\n"
                    f"OWNER:       {v['owner_name']} ({v['owner_role'].upper()})\n"
                    f"DEPARTMENT:  {v['department']}\n"
                    f"PASS_ID:     {v['pass_sticker_id']}\n"
                    f"CONTACT:     {v['contact_phone']}",
                    language="yaml",
                )
            else:
                st.code(
                    f"STATUS:      UNREGISTERED_OUTSIDER_ALERT\n"
                    f"PLATE_NO:    {result['plate_number']}\n"
                    f"DIRECTIVE:   Hold gate barrier; log temporary visitor token.",
                    language="yaml",
                )

        st.markdown('<div class="sec-label">03.B // ENROLL NEW CAMPUS VEHICLE PASS</div>', unsafe_allow_html=True)
        with st.form("new_vehicle_form"):
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                p_num = st.text_input("Plate Number", placeholder="MH20GH7788")
                o_role = st.selectbox("Role", ["student", "faculty", "staff", "college_bus", "ambulance"])
                v_type = st.selectbox("Class", ["two_wheeler", "four_wheeler", "bus", "emergency"])
            with f_c2:
                o_name = st.text_input("Full Name", placeholder="Rohan Deshmukh")
                dept = st.text_input("Department", placeholder="CSE / AI&DS / Mech")
                sticker = st.text_input("Pass Sticker ID", placeholder="CSMSS-PASS-2026-501")
            phone = st.text_input("Contact Number", placeholder="9823000000")
            reg_submit = st.form_submit_button("COMMIT TO WHITELIST", use_container_width=True)

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
                    st.rerun()
                except Exception as exc:
                    st.error(f"Enrollment error: {exc}")

    with col_reg:
        st.markdown('<div class="sec-label">03.C // AUTHORIZED VEHICLE REGISTRY (SQLITE)</div>', unsafe_allow_html=True)
        vehicles = list_registered_vehicles()
        if vehicles:
            st.dataframe(
                pd.DataFrame(vehicles)[
                    [
                        "plate_number",
                        "owner_name",
                        "owner_role",
                        "department",
                        "vehicle_type",
                        "pass_sticker_id",
                        "contact_phone",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "plate_number": "Plate No.",
                    "owner_name": "Registered Owner",
                    "owner_role": "Role",
                    "department": "Dept",
                    "vehicle_type": "Class",
                    "pass_sticker_id": "Sticker ID",
                    "contact_phone": "Contact",
                },
            )