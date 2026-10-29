"""
Tab 3 — Road Surface Health & Civil Work Orders (`frontend/views/road_health.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering.

Features:
1. Expandable "Methodology & Physics Note" card explaining:
   - Perspective Homography Surface Area ($m^2$) calibration (`pixel_area / (px_per_meter ** 2)`)
   - Composite 0–100 Road Damage Severity Index formula (50% Normalized Cavity Area +
     30% Interior Shadow Darkness Ratio + 20% Perimeter-to-Area Edge Jaggedness)
   - Heuristic Shadow-Intensity Depth Proxy (cm) & Civil Repair Cost estimation in INR.
2. Zone-wise road quality table and logged pothole telemetry.
3. 1-Click "Generate Civil Maintenance Work Order Summary" view with downloadable report.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from config.settings import (
    BASE_REPAIR_RATE_INR_PER_SQM,
    CAMPUS_ZONES,
    INSTITUTION_NAME,
    MIN_REPAIR_MOBILIZATION_COST_INR,
    PIXELS_PER_METER_BY_ZONE,
)

IST = timezone(timedelta(hours=5, minutes=30))


def render_road_health_tab(
    road_logs: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
    active_role_label: str,
) -> None:
    """Render Tab 3: Road Surface Health, Physics Methodology Card, and Civil Work Order Generator."""
    st.subheader("🛣️ Calibrated Road Surface Health & Civil Maintenance Engineering")
    st.caption(
        "Monocular computer vision road surface assessment calibrated for CSMSS Kanchanwadi campus corridors."
    )

    # ------------------------------------------------------------------
    # 1. Expandable Methodology & Physics Note (Designed for Viva Examiners)
    # ------------------------------------------------------------------
    with st.expander("📐 Methodology & Physics Note: Homography Area (m²) & Severity Index (0–100) Formulation", expanded=True):
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                """
                #### 1. Perspective Homography Surface Area ($m^2$)
                Each campus camera has a calibrated homography scale constant $k_{\\text{zone}}$
                (`px_per_meter`) derived from mounting height and road-plane fiducial markers:
                $$A_{\\text{sqm}} = \\frac{A_{\\text{pixels}}}{\\left(k_{\\text{zone}}\\right)^2}$$
                - **`ZONE-01` (Main Gate):** $92.0\\text{ px/m}$ ($4.2\\text{ m}$ gantry)
                - **`ZONE-02` (Diploma Circle):** $84.5\\text{ px/m}$ ($4.8\\text{ m}$ pole)
                - **`ZONE-03` (Central Library Road):** $78.0\\text{ px/m}$ ($5.0\\text{ m}$ corridor)
                - **`ZONE-04` (Workshop Bay):** $96.0\\text{ px/m}$ ($3.8\\text{ m}$ bay mount)
                """
            )
        with m_col2:
            st.markdown(
                f"""
                #### 2. Defensible Road Damage Severity Index ($0\\text{{--}}100$)
                Combines three normalized physical and photometric indicators:
                $$\\text{{SI}}_{{0-100}} = 100 \\times \\Big(0.50\\,\\hat{{A}} + 0.30\\,\\hat{{D}}_{{\\text{{shadow}}}} + 0.20\\,\\hat{{J}}_{{\\text{{edge}}}}\\Big)$$
                - **Normalized Cavity Area ($\\hat{{A}}$, 50% weight):** $\\min(1.0,\\, A_{{\\text{{sqm}}}} / 1.45\\text{{ m}}^2)$
                - **Interior Shadow Darkness Ratio ($\\hat{{D}}_{{\\text{{shadow}}}}$, 30% weight):** Relative HSV $V$-channel luminance drop inside the cavity vs. surrounding asphalt median.
                - **Edge Jaggedness ($\\hat{{J}}_{{\\text{{edge}}}}$, 20% weight):** Isoperimetric quotient inverse $P^2 / (4\\pi A_{{\\text{{pixels}}}})$ capturing fractured asphalt rims.
                - **Civil Repair Cost (INR):** $\\text{{INR }}\\,{MIN_REPAIR_MOBILIZATION_COST_INR:.0f} + A_{{\\text{{sqm}}}} \\times \\text{{INR }}\\,{BASE_REPAIR_RATE_INR_PER_SQM:.0f} \\times (1 + 0.85 \\cdot \\frac{{\\text{{SI}}}}{{100}}) \\times w_{{\\text{{zone}}}}$
                """
            )

    # ------------------------------------------------------------------
    # 2. Zone-Wise Road Quality & Structural Health Table
    # ------------------------------------------------------------------
    st.markdown("### 📋 Zone-Wise Road Surface Quality Scorecard")
    zone_summary_rows: List[Dict[str, Any]] = []

    for z in zones:
        zid = z["zone_id"]
        z_logs = [r for r in road_logs if r.get("zone_id") == zid]
        cav_cnt = len(z_logs)
        tot_area = round(sum(float(r.get("surface_area_sqm", 0.0)) for r in z_logs), 3)
        max_si = round(max([float(r.get("severity_index", 0.0)) for r in z_logs], default=0.0), 1)
        avg_depth = (
            round(sum(float(r.get("estimated_depth_proxy_cm", 0.0)) for r in z_logs) / cav_cnt, 2)
            if cav_cnt > 0
            else 0.0
        )
        tot_cost = round(sum(float(r.get("estimated_repair_cost_inr", 0.0)) for r in z_logs), 2)

        if max_si >= 75.0:
            surface_grade = "D — Immediate Cold-Mix / Hot-Mix Patch Required"
        elif max_si >= 45.0:
            surface_grade = "C — Moderate Ravelling (Schedule Maintenance)"
        elif cav_cnt > 0:
            surface_grade = "B — Minor Surface Crack (Preventive Sealant)"
        else:
            surface_grade = "A — Good Structural Integrity"

        zone_summary_rows.append(
            {
                "Zone ID": zid,
                "Campus Corridor": z["zone_name"],
                "Calibration (px/m)": PIXELS_PER_METER_BY_ZONE.get(zid, z["px_per_meter"]),
                "Logged Cavities": cav_cnt,
                "Damaged Area (m²)": tot_area,
                "Peak Severity Index (0-100)": max_si,
                "Avg Shadow Depth Proxy (cm)": avg_depth,
                "Est. Repair Budget (INR)": f"₹{tot_cost:,.2f}",
                "Civil Condition Grade": surface_grade,
            }
        )

    st.dataframe(pd.DataFrame(zone_summary_rows), use_container_width=True, hide_index=True)

    # Detailed Pothole Telemetry Logs
    st.markdown("### 🕳️ Calibrated Pothole & Cavity Telemetry Logs")
    if road_logs:
        log_rows = []
        for r in road_logs:
            log_rows.append(
                {
                    "Log ID": f"RD-{r['log_id']:03d}",
                    "Incident ID": f"#{r['incident_id']}",
                    "Zone": f"{r.get('zone_id')} ({r.get('zone_name', '')})",
                    "Pothole Class": r.get("pothole_class"),
                    "Surface Area (m²)": round(float(r.get("surface_area_sqm", 0.0)), 3),
                    "Severity Index (0-100)": round(float(r.get("severity_index", 0.0)), 1),
                    "Shadow Depth Proxy (cm)": round(float(r.get("estimated_depth_proxy_cm", 0.0)), 2),
                    "Depth Method": r.get(
                        "depth_estimation_method", "Heuristic Shadow-Intensity Proxy (Monocular CV)"
                    ),
                    "Est. Repair Cost (INR)": f"₹{float(r.get('estimated_repair_cost_inr', 0.0)):,.2f}",
                    "Status": r.get("status", "Open"),
                }
            )
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No road damage logs currently recorded.")

    st.markdown("<hr style='border-color: rgba(148,163,184,0.18); margin: 14px 0;'/>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. 1-Click "Generate Civil Maintenance Work Order Summary" View
    # ------------------------------------------------------------------
    st.markdown("### 🧾 Civil Maintenance Work Order Generator")
    if "show_civil_work_order" not in st.session_state:
        st.session_state["show_civil_work_order"] = False

    if st.button(
        "🛠️ Generate Civil Maintenance Work Order Summary",
        type="primary",
        key="btn_generate_work_order",
    ):
        st.session_state["show_civil_work_order"] = True

    if st.session_state["show_civil_work_order"] and road_logs:
        active_road_logs = [r for r in road_logs if r.get("status") != "Resolved"] or road_logs
        total_area = sum(float(r.get("surface_area_sqm", 0.0)) for r in active_road_logs)
        total_budget = sum(float(r.get("estimated_repair_cost_inr", 0.0)) for r in active_road_logs)
        bitumen_kg = round(total_area * 115.0, 1)
        aggregate_kg = round(total_area * 240.0, 1)
        stamp_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")

        work_order_md = f"""
### 🏛️ CIVIL ESTATE WORK ORDER — {INSTITUTION_NAME}
- **Work Order Reference:** `CSMSS/CIVIL/WO-2026-1025`
- **Generated Timestamp:** `{stamp_ist}`
- **Prepared Role:** `{active_role_label}`
- **Active Road Hazards Queued:** `{len(active_road_logs)} locations`
- **Total Calibrated Damaged Surface Area:** `{total_area:.2f} m²`
- **Estimated Material Requirement:** `{bitumen_kg} kg` Cold/Hot Bitumen Mix + `{aggregate_kg} kg` Graded Stone Aggregate
- **Total Sanctioned Repair Estimate:** **`INR ₹{total_budget:,.2f}`**

| Priority | Log ID | Campus Zone | Pothole Class | Calibrated Area ($m^2$) | Severity Index | Depth Proxy (cm) | Est. Cost (INR) |
| :-: | :--- | :--- | :--- | --: | --: | --: | --: |
"""
        for idx, r in enumerate(active_road_logs, start=1):
            work_order_md += (
                f"| **P{idx}** | `RD-{r['log_id']:03d}` (Inc `#{r['incident_id']}`) | "
                f"`{r.get('zone_id')}` {r.get('zone_name', '')} | **{r.get('pothole_class')}** | "
                f"`{float(r.get('surface_area_sqm', 0.0)):.2f} m²` | "
                f"`{float(r.get('severity_index', 0.0)):.1f}/100` | "
                f"`{float(r.get('estimated_depth_proxy_cm', 0.0)):.1f} cm` | "
                f"**₹{float(r.get('estimated_repair_cost_inr', 0.0)):,.2f}** |\n"
            )

        st.markdown(
            f'<div class="csmss-section-card">{work_order_md}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="📥 Download Civil Maintenance Work Order (.md)",
            data=work_order_md,
            file_name="CSMSS_Kanchanwadi_Civil_Work_Order.md",
            mime="text/markdown",
            key="dl_civil_work_order_btn",
        )