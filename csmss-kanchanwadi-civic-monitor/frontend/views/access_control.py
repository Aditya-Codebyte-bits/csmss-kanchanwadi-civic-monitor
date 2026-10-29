"""
Tab 4 — ANPR Access Control & Privacy-Aware Registry (`frontend/views/access_control.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering.

Features:
1. Searchable campus vehicle database (`MH 20` Chhatrapati Sambhajinagar plates)
   with role and whitelist status filters.
2. Automatic privacy-by-design phone number masking (`+91 98XXX-XX412` or `[REDACTED]`)
   when viewed in `Student / Public Kiosk (Privacy Masked)` mode.
3. Quick Vehicle Registration form (`POST /vehicles` or direct SQLite upsert) with
   live Indian RTO regex validation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pandas as pd
import streamlit as st

from config.settings import VALID_USER_ROLES, VALID_VEHICLE_TYPES
from cv_engine.anpr_reader import validate_and_normalize_indian_plate
from frontend.styles import ROLE_TO_INTERNAL, format_privacy_contact


def render_access_control_tab(
    vehicles: List[Dict[str, Any]],
    active_role_label: str,
    on_register_vehicle: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Render Tab 4: ANPR Access Control & Privacy-Aware Vehicle Registry."""
    internal_role = ROLE_TO_INTERNAL.get(active_role_label, "Student")
    is_privacy_masked = internal_role not in {"Campus Admin", "Security Guard"}

    st.subheader("🚘 ANPR Campus Access Control & Privacy-Aware Vehicle Registry")
    if is_privacy_masked:
        st.warning(
            f"🔒 **Privacy-by-Design Active (`{active_role_label}`):** Personal phone numbers are "
            "automatically masked (`+91 98XXX-XX412` / `[REDACTED]`) to protect student and faculty PII. "
            "Switch `View As` to `Campus Director / Examiner` or `Security Control Room` to view unmasked records."
        )
    else:
        st.success(
            f"🔓 **Authorized Enforcement View (`{active_role_label}`):** Full unmasked contact telemetry enabled."
        )

    # ------------------------------------------------------------------
    # 1. Searchable Vehicle Registry Table
    # ------------------------------------------------------------------
    s_col1, s_col2, s_col3, s_col4 = st.columns([1.5, 1.0, 1.0, 1.0])
    with s_col1:
        search_query = st.text_input(
            "Search by Plate Number (`MH 20 ...`) or Owner Name",
            value="",
            placeholder="e.g. MH20, Deshmukh, Bus, 4501...",
            key="ac_search_input",
        )
    with s_col2:
        role_filter = st.selectbox(
            "Filter by Owner Role",
            options=["ALL"] + list(VALID_USER_ROLES),
            key="ac_role_filter",
        )
    with s_col3:
        vtype_filter = st.selectbox(
            "Filter by Vehicle Class",
            options=["ALL"] + list(VALID_VEHICLE_TYPES),
            key="ac_vtype_filter",
        )
    with s_col4:
        mask_mode = st.selectbox(
            "Public Kiosk Masking Style",
            options=["Pattern (+91 98XXX-XX412)", "Strict ([REDACTED])"],
            index=0,
            disabled=not is_privacy_masked,
            key="ac_mask_style",
        )

    redact_strictly = mask_mode.startswith("Strict")

    filtered_vehicles = []
    q_norm = search_query.strip().lower()
    for v in vehicles:
        if role_filter != "ALL" and v.get("role") != role_filter:
            continue
        if vtype_filter != "ALL" and v.get("vehicle_type") != vtype_filter:
            continue
        if q_norm:
            haystack = f"{v.get('plate_number', '')} {v.get('owner_name', '')} {v.get('vehicle_type', '')}".lower()
            if q_norm not in haystack:
                continue
        filtered_vehicles.append(v)

    table_rows = []
    for v in filtered_vehicles:
        _, formatted_plate, _ = validate_and_normalize_indian_plate(v.get("plate_number", ""))
        display_contact = format_privacy_contact(
            v.get("contact_no", ""),
            viewer_label=active_role_label,
            redact_completely=redact_strictly,
        )
        table_rows.append(
            {
                "ID": f"VEH-{v.get('vehicle_id', 0):03d}",
                "RTO Plate Number": formatted_plate or v.get("plate_number"),
                "Compact Key": v.get("plate_number"),
                "Owner Name": v.get("owner_name"),
                "Campus Role": v.get("role"),
                "Vehicle Type": v.get("vehicle_type"),
                "Contact Number (RBAC)": display_contact,
                "Gate Access Status": "✅ Whitelisted" if v.get("is_whitelisted") else "🚫 Restricted / Alert",
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: rgba(148,163,184,0.18); margin: 16px 0;'/>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2. Quick Vehicle Registration Form
    # ------------------------------------------------------------------
    st.markdown("### ➕ Quick Campus Vehicle Registration / Whitelist Update")
    st.caption(
        "Register a new student, faculty, bus, or emergency vehicle into `registered_vehicles` "
        "with automatic Indian RTO (`MH 20`) normalization."
    )

    with st.form("quick_vehicle_registration_form", clear_on_submit=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            new_plate = st.text_input(
                "Indian RTO License Plate*",
                value="MH 20 EE 7890",
                help="Standard Indian RTO format, e.g. MH 20 EE 7890",
            )
            new_owner = st.text_input(
                "Owner Full Name*",
                value="Prof. Anjali Deshpande",
            )
        with f2:
            new_role = st.selectbox(
                "Campus Role*",
                options=list(VALID_USER_ROLES),
                index=1,
            )
            new_vtype = st.selectbox(
                "Vehicle Category*",
                options=list(VALID_VEHICLE_TYPES),
                index=1,
            )
        with f3:
            new_contact = st.text_input(
                "Mobile Contact Number*",
                value="+91-9823456712",
            )
            new_whitelist = st.checkbox(
                "Grant Campus Gate Whitelist Authorization",
                value=True,
            )

        submitted = st.form_submit_button("Register / Update Campus Vehicle", type="primary")
        if submitted:
            compact_p, fmt_p, is_valid_rto = validate_and_normalize_indian_plate(new_plate)
            if not is_valid_rto or not compact_p:
                st.error(
                    f"Invalid Indian RTO license plate `{new_plate}`. Expected format like `MH 20 EE 4501`."
                )
            elif len(new_owner.strip()) < 2:
                st.error("Please provide a valid owner name.")
            else:
                saved = on_register_vehicle(
                    {
                        "plate_number": compact_p,
                        "owner_name": new_owner.strip(),
                        "role": new_role,
                        "vehicle_type": new_vtype,
                        "contact_no": new_contact.strip(),
                        "is_whitelisted": bool(new_whitelist),
                    }
                )
                st.toast(f"Vehicle {fmt_p} registered for {saved.get('owner_name')}!", icon="🚘")
                st.success(
                    f"Successfully saved **{fmt_p}** (`{compact_p}`) under **{saved.get('owner_name')}** "
                    f"({'Whitelisted' if saved.get('is_whitelisted') else 'Restricted'})."
                )
                st.rerun()