"""
Tab 1 — Command Center & 3D Geospatial Map (`frontend/views/command_center.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering.

Features:
1. 5 KPI Cards with delta badges:
   - `Active Open Incidents`
   - `Deduplicated Spam Suppressed`
   - `Avg Road Severity Index`
   - `Whitelisted Fleet Ratio`
   - `Avg Inference Speed (ms)`
2. Interactive 3D `pydeck` map centered on CSMSS Kanchanwadi (`19.8260, 75.2950`)
   with color-coded 3D columns + scatter points per zone height-scaled by incident
   criticality, plus a clean fallback 2D coordinate plot if WebGL is unavailable.
3. Live Incident Action Feed where Security/Maintenance/Admin roles can filter
   incidents and transition status (`Open → Investigating → Resolved`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import streamlit as st

from config.settings import CAMPUS_CENTER_LAT, CAMPUS_CENTER_LON, CAMPUS_ZONES, UI_COLOR_TOKENS
from frontend.styles import ROLE_TO_INTERNAL, render_kpi_card, render_severity_badge


def _build_zone_geospatial_dataframe(
    zones: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate active incidents and criticality scores per CSMSS campus zone for 3D mapping."""
    zone_rows: List[Dict[str, Any]] = []
    sev_weights = {"Low": 1, "Medium": 2, "High": 4, "Critical": 7}

    for z in zones:
        zid = z["zone_id"]
        z_incs = [i for i in incidents if i.get("zone_id") == zid]
        open_incs = [i for i in z_incs if i.get("status") != "Resolved"]
        crit_cnt = sum(1 for i in open_incs if i.get("severity") == "Critical")
        repeat_sum = sum(int(i.get("repeat_count", 1)) for i in z_incs)

        hazard_score = sum(sev_weights.get(i.get("severity", "Low"), 1) for i in open_incs)
        elevation_m = max(18.0, float(hazard_score) * float(z.get("criticality_weight", 1.2)) * 12.0)

        rgb = list(UI_COLOR_TOKENS["zone_rgb"].get(zid, [14, 165, 233, 210]))
        if crit_cnt > 0:
            rgb = [239, 68, 68, 225]
        elif len(open_incs) > 0:
            rgb = [245, 158, 11, 215]
        else:
            rgb = [16, 185, 129, 210]

        zone_rows.append(
            {
                "zone_id": zid,
                "zone_name": z["zone_name"],
                "latitude": float(z["latitude"]),
                "longitude": float(z["longitude"]),
                "criticality_weight": float(z["criticality_weight"]),
                "px_per_meter": float(z["px_per_meter"]),
                "total_incidents": len(z_incs),
                "open_incidents": len(open_incs),
                "critical_incidents": crit_cnt,
                "repeat_events": repeat_sum,
                "elevation_m": round(elevation_m, 1),
                "color_r": rgb[0],
                "color_g": rgb[1],
                "color_b": rgb[2],
                "color_a": rgb[3],
            }
        )
    return pd.DataFrame(zone_rows)


def _render_fallback_2d_zone_plot(zone_df: pd.DataFrame) -> None:
    """Render a clean dark-themed 2D Matplotlib coordinate map if WebGL is disabled or desired."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    for _, row in zone_df.iterrows():
        col_hex = (
            "#EF4444"
            if row["criticality_weight"] >= 1.4 or row["critical_incidents"] > 0
            else "#F59E0B"
            if row["open_incidents"] > 0
            else "#10B981"
        )
        size = 140 + row["elevation_m"] * 1.4
        ax.scatter(
            row["longitude"],
            row["latitude"],
            s=size,
            c=col_hex,
            alpha=0.85,
            edgecolors="#F8FAFC",
            linewidth=1.5,
        )
        ax.annotate(
            f"{row['zone_id']}: {row['zone_name']}\n(Active: {row['open_incidents']}, Wt: {row['criticality_weight']}x)",
            (row["longitude"], row["latitude"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=7.5,
            color="#F8FAFC",
            weight="bold",
        )

    ax.set_title(
        "CSMSS Kanchanwadi Campus — 2D Geospatial Zone Coordinate Plot",
        color="#F8FAFC",
        fontsize=9.5,
        pad=10,
    )
    ax.set_xlabel("Longitude (°E)", color="#94A3B8", fontsize=8)
    ax.set_ylabel("Latitude (°N)", color="#94A3B8", fontsize=8)
    ax.tick_params(colors="#CBD5E1", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748B")
    # Expand margins slightly so labels fit cleanly
    ax.margins(0.28)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_command_center_tab(
    summary: Dict[str, Any],
    zones: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
    road_logs: List[Dict[str, Any]],
    avg_latency_ms: float,
    active_role_label: str,
    on_status_update: Callable[[int, str, Optional[str]], bool],
) -> None:
    """Render Tab 1: Command Center KPIs, 3D Geospatial Map, and Live Incident Action Feed."""
    internal_role = ROLE_TO_INTERNAL.get(active_role_label, "Student")
    can_mutate_status = internal_role in {
        "Campus Admin",
        "Security Guard",
        "Civil Maintenance Staff",
    }

    # ------------------------------------------------------------------
    # 1. Top Row: 5 KPI Cards with Delta Badges
    # ------------------------------------------------------------------
    open_cnt = int(summary.get("open_incidents", 0))
    inv_cnt = int(summary.get("investigating_incidents", 0))
    crit_cnt = int(summary.get("critical_incidents", 0))
    dedup_saved = int(summary.get("deduplication_savings_count", 0))
    dedup_eff = float(summary.get("deduplication_efficiency_pct", 0.0))

    avg_road_si = (
        round(float(sum(float(r.get("severity_index", 0.0)) for r in road_logs)) / len(road_logs), 1)
        if road_logs
        else 0.0
    )
    total_veh = max(1, int(summary.get("registered_vehicles_count", 1)))
    white_veh = int(summary.get("whitelisted_vehicles_count", 0))
    fleet_ratio_pct = round((white_veh / total_veh) * 100.0, 1)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card(
            label="Active Open Incidents",
            value=f"{open_cnt + inv_cnt}",
            delta_text=f"{crit_cnt} Critical • {open_cnt} Open / {inv_cnt} Inv",
            accent_hex="#EF4444" if crit_cnt > 0 else "#F59E0B",
            delta_bg_rgba="rgba(239, 68, 68, 0.2)" if crit_cnt > 0 else "rgba(245, 158, 11, 0.2)",
            delta_text_hex="#FCA5A5" if crit_cnt > 0 else "#FDE68A",
        )
    with k2:
        render_kpi_card(
            label="Deduplicated Spam Suppressed",
            value=f"{dedup_saved} Frames",
            delta_text=f"{dedup_eff:.1f}% DB Write Reduction (15m Window)",
            accent_hex="#0284C7",
            delta_bg_rgba="rgba(2, 132, 199, 0.22)",
            delta_text_hex="#38BDF8",
        )
    with k3:
        render_kpi_card(
            label="Avg Road Severity Index",
            value=f"{avg_road_si:.1f} / 100",
            delta_text=f"{summary.get('total_damaged_area_sqm', 0.0):.2f} m² • INR {summary.get('total_estimated_repair_cost_inr', 0.0):,.0f}",
            accent_hex="#F97316",
            delta_bg_rgba="rgba(249, 115, 22, 0.22)",
            delta_text_hex="#FDBA74",
        )
    with k4:
        render_kpi_card(
            label="Whitelisted Fleet Ratio",
            value=f"{fleet_ratio_pct:.1f}%",
            delta_text=f"{white_veh} / {total_veh} MH-20 Campus Vehicles",
            accent_hex="#10B981",
            delta_bg_rgba="rgba(16, 185, 129, 0.22)",
            delta_text_hex="#6EE7B7",
        )
    with k5:
        render_kpi_card(
            label="Avg Inference Speed (ms)",
            value=f"{avg_latency_ms:.1f} ms",
            delta_text="Target < 250 ms CPU Real-Time",
            accent_hex="#8B5CF6",
            delta_bg_rgba="rgba(139, 92, 246, 0.22)",
            delta_text_hex="#C4B5FD",
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2. Middle Split: 3D PyDeck Campus Map + Zone Breakdown Telemetry
    # ------------------------------------------------------------------
    zone_df = _build_zone_geospatial_dataframe(zones=zones, incidents=incidents)

    col_map, col_stats = st.columns([1.45, 1.0])
    with col_map:
        st.subheader("🗺️ CSMSS Kanchanwadi 3D Geospatial Telemetry Map")
        st.caption(
            f"Centered on CSMSS Chh. Shahu College of Engineering (`{CAMPUS_CENTER_LAT:.4f}°N, {CAMPUS_CENTER_LON:.4f}°E`) "
            "• Column height scaled by active incident criticality × zone weight."
        )

        show_2d_fallback = st.checkbox(
            "Show 2D Coordinate Plot (WebGL-Free Fallback View)",
            value=False,
            key="cmd_2d_fallback_toggle",
        )

        if not show_2d_fallback:
            try:
                col_layer = pdk.Layer(
                    "ColumnLayer",
                    data=zone_df,
                    get_position=["longitude", "latitude"],
                    get_elevation="elevation_m",
                    elevation_scale=1.2,
                    radius=22,
                    get_fill_color=["color_r", "color_g", "color_b", "color_a"],
                    pickable=True,
                    auto_highlight=True,
                    extruded=True,
                )
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=zone_df,
                    get_position=["longitude", "latitude"],
                    get_radius=36,
                    get_fill_color=[56, 189, 248, 90],
                    get_line_color=[248, 250, 252, 200],
                    stroked=True,
                    line_width_min_pixels=2,
                    pickable=True,
                )
                text_layer = pdk.Layer(
                    "TextLayer",
                    data=zone_df,
                    get_position=["longitude", "latitude"],
                    get_text="zone_id",
                    get_color=[248, 250, 252, 255],
                    get_size=15,
                    get_alignment_baseline="'bottom'",
                )
                view_state = pdk.ViewState(
                    latitude=CAMPUS_CENTER_LAT,
                    longitude=CAMPUS_CENTER_LON,
                    zoom=16.6,
                    pitch=52,
                    bearing=-18,
                )
                tooltip = {
                    "html": (
                        "<b>{zone_id}: {zone_name}</b><br/>"
                        "Active Open/Inv: <b>{open_incidents}</b> (Critical: {critical_incidents})<br/>"
                        "Total Historical: {total_incidents} | Repeat Frames: {repeat_events}<br/>"
                        "Criticality Weight: {criticality_weight}x | Scale: {px_per_meter} px/m"
                    ),
                    "style": {
                        "backgroundColor": "#0F172A",
                        "color": "#F8FAFC",
                        "fontSize": "12px",
                        "border": "1px solid #38BDF8",
                        "borderRadius": "8px",
                        "padding": "8px",
                    },
                }
                deck = pdk.Deck(
                    layers=[scatter_layer, col_layer, text_layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                )
                st.pydeck_chart(deck, use_container_width=True)
            except Exception:
                _render_fallback_2d_zone_plot(zone_df)
        else:
            _render_fallback_2d_zone_plot(zone_df)

    with col_stats:
        st.subheader("📊 Campus Zone Criticality Matrix")
        st.caption("Real-time telemetry across the 4 monitored CSMSS Kanchanwadi corridors.")
        display_zone_df = zone_df[
            [
                "zone_id",
                "zone_name",
                "criticality_weight",
                "open_incidents",
                "critical_incidents",
                "repeat_events",
            ]
        ].rename(
            columns={
                "zone_id": "Zone",
                "zone_name": "Corridor Name",
                "criticality_weight": "Weight",
                "open_incidents": "Active",
                "critical_incidents": "Critical",
                "repeat_events": "Total Detections",
            }
        )
        st.dataframe(display_zone_df, use_container_width=True, hide_index=True)

        sev_dist = summary.get("severity_distribution", {})
        type_dist = summary.get("incident_type_distribution", {})
        c_sev, c_type = st.columns(2)
        with c_sev:
            st.markdown("**Severity Breakdown**")
            for sev_key in ("Critical", "High", "Medium", "Low"):
                cnt = int(sev_dist.get(sev_key, 0))
                st.markdown(
                    f"{render_severity_badge(sev_key)} &nbsp; **{cnt}** incident(s)",
                    unsafe_allow_html=True,
                )
        with c_type:
            st.markdown("**Incident Categories**")
            for t_name, t_cnt in type_dist.items():
                st.markdown(f"- **{t_name}**: `{t_cnt}`")

    st.markdown("<hr style='border-color: rgba(148,163,184,0.18); margin: 16px 0;'/>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. Live Incident Action Feed & Status Transition Console
    # ------------------------------------------------------------------
    st.subheader("🚨 Live Campus Incident Action Feed & Dispatch Queue")

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.1, 1.1, 1.2, 1.2])
    with f_col1:
        zone_filter = st.selectbox(
            "Filter by Campus Zone",
            options=["ALL", "ZONE-01", "ZONE-02", "ZONE-03", "ZONE-04"],
            key="cmd_zone_filter",
        )
    with f_col2:
        status_filter = st.selectbox(
            "Filter by Lifecycle Status",
            options=["ALL", "Open", "Investigating", "Resolved"],
            key="cmd_status_filter",
        )
    with f_col3:
        type_filter = st.selectbox(
            "Filter by Incident Type",
            options=[
                "ALL",
                "Emergency Lane Blocked",
                "Unlawful Parking",
                "Road Damage",
                "Unauthorized Vehicle",
            ],
            key="cmd_type_filter",
        )
    with f_col4:
        sev_filter = st.selectbox(
            "Filter by Severity",
            options=["ALL", "Critical", "High", "Medium", "Low"],
            key="cmd_sev_filter",
        )

    filtered_incidents = [
        inc
        for inc in incidents
        if (zone_filter == "ALL" or inc.get("zone_id") == zone_filter)
        and (status_filter == "ALL" or inc.get("status") == status_filter)
        and (type_filter == "ALL" or inc.get("incident_type") == type_filter)
        and (sev_filter == "ALL" or inc.get("severity") == sev_filter)
    ]

    if not filtered_incidents:
        st.info("No campus incidents match the current filter criteria.")
        return

    # Build clean tabular overview
    feed_rows = []
    for inc in filtered_incidents[:25]:
        plate_val = inc.get("plate_number") or "N/A (Road Surface)"
        if internal_role == "Student" and plate_val != "N/A (Road Surface)":
            # Mask middle characters in public kiosk mode if desired, or keep RTO prefix visible
            plate_val = f"{plate_val[:4]}-XX-{plate_val[-4:]}" if len(plate_val) >= 8 else plate_val

        feed_rows.append(
            {
                "ID": f"#{inc['incident_id']}",
                "Last Seen (IST)": inc.get("last_seen_timestamp", ""),
                "Zone": f"{inc['zone_id']} ({inc.get('zone_name') or CAMPUS_ZONES.get(inc['zone_id'], {}).get('zone_name', '')})",
                "Incident Type": inc.get("incident_type"),
                "Plate / Target": plate_val,
                "Severity": inc.get("severity"),
                "Status": inc.get("status"),
                "Dedup Count (x)": int(inc.get("repeat_count", 1)),
                "Confidence": f"{float(inc.get('confidence_score', 0.88)) * 100:.1f}%",
                "Details": inc.get("details", ""),
            }
        )
    st.dataframe(pd.DataFrame(feed_rows), use_container_width=True, hide_index=True)

    # Interactive Status Transition Bar for Authorized Roles
    if can_mutate_status:
        with st.expander("🛠️ Dispatch & Incident Lifecycle Transition Console (Open → Investigating → Resolved)", expanded=True):
            t_col1, t_col2, t_col3, t_col4 = st.columns([1.2, 1.0, 1.8, 0.9])
            inc_options = {
                f"#{inc['incident_id']} | {inc['zone_id']} | {inc['incident_type']} ({inc['status']})": int(
                    inc["incident_id"]
                )
                for inc in filtered_incidents
            }
            with t_col1:
                selected_label = st.selectbox(
                    "Select Incident ID",
                    options=list(inc_options.keys()),
                    key="cmd_transition_inc_select",
                )
            with t_col2:
                next_status = st.selectbox(
                    "New Status",
                    options=["Investigating", "Resolved", "Open"],
                    key="cmd_transition_next_status",
                )
            with t_col3:
                notes = st.text_input(
                    "Dispatch / Resolution Note",
                    value=f"Updated by {active_role_label.split('(')[0].strip()}",
                    key="cmd_transition_notes",
                )
            with t_col4:
                st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
                if st.button("Apply Status Update", type="primary", use_container_width=True):
                    target_id = inc_options[selected_label]
                    ok = on_status_update(target_id, next_status, notes)
                    if ok:
                        st.toast(f"Incident #{target_id} transitioned to {next_status}.", icon="✅")
                        st.rerun()
    else:
        st.caption(
            "🔒 **Read-Only Kiosk Mode**: Switch `View As` in the top bar to `Security Control Room`, "
            "`Civil Maintenance Engineer`, or `Campus Director / Examiner` to dispatch or resolve incidents."
        )