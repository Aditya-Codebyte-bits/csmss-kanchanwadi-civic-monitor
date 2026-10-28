"""
UI Theme, Custom CSS & Reusable HTML Components (`frontend/styles.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering,
Kanchanwadi, Chhatrapati Sambhajinagar.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import streamlit as st

from config.settings import APP_TITLE, APP_VERSION, INSTITUTION_NAME, UI_COLOR_TOKENS

# Canonical Role Switcher options required for Viva Demonstration
ROLE_VIEW_OPTIONS = (
    "Campus Director / Examiner (Full Telemetry)",
    "Security Control Room",
    "Civil Maintenance Engineer",
    "Student / Public Kiosk (Privacy Masked)",
)

ROLE_TO_INTERNAL: Dict[str, str] = {
    "Campus Director / Examiner (Full Telemetry)": "Campus Admin",
    "Security Control Room": "Security Guard",
    "Civil Maintenance Engineer": "Civil Maintenance Staff",
    "Student / Public Kiosk (Privacy Masked)": "Student",
}


def format_privacy_contact(
    contact_no: str,
    viewer_label: str,
    redact_completely: bool = False,
) -> str:
    """
    Privacy-by-Design Phone Number Formatter:
    - Full Telemetry (`Campus Director / Examiner`, `Security Control Room`):
      returns unmasked phone number (e.g. `+91-9823014522`).
    - `Civil Maintenance Engineer`:
      returns partially masked number (`+91 98XXX-XX522`).
    - `Student / Public Kiosk (Privacy Masked)`:
      returns `+91 98XXX-XX412` pattern (e.g. `+91 98XXX-XX876`) or `[REDACTED]`
      when `redact_completely=True`.
    """
    if not contact_no:
        return "N/A"
    internal_role = ROLE_TO_INTERNAL.get(viewer_label, viewer_label)
    if internal_role in {"Campus Admin", "Security Guard"}:
        return contact_no

    if redact_completely and internal_role == "Student":
        return "[REDACTED]"

    digits = re.sub(r"\D", "", contact_no)
    if len(digits) >= 10:
        local10 = digits[-10:]
        first2 = local10[:2]
        last3 = local10[-3:]
        return f"+91 {first2}XXX-XX{last3}"
    return "[REDACTED]"


def apply_custom_styles() -> None:
    """Inject production-grade dark tactical CSS for CSMSS Kanchanwadi Civic Monitor v2.0."""
    brand = UI_COLOR_TOKENS["brand"]
    st.markdown(
        f"""
        <style>
        /* Global container spacing */
        .block-container {{
            padding-top: 1.1rem !important;
            padding-bottom: 2.0rem !important;
            max-width: 1440px !important;
        }}

        /* Top Persistent System Header Banner */
        .csmss-top-header {{
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #1E293B 100%);
            border: 1px solid rgba(56, 189, 248, 0.32);
            border-left: 6px solid {brand['secondary']};
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        }}
        .csmss-title-row {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }}
        .csmss-main-title {{
            font-size: 1.42rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: 0.3px;
            margin: 0;
        }}
        .csmss-sub-title {{
            font-size: 0.84rem;
            color: #CBD5E1;
            margin-top: 3px;
        }}
        .csmss-pill-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }}
        .csmss-pill {{
            display: inline-flex;
            align-items: center;
            padding: 4px 11px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.25px;
            border: 1px solid rgba(255,255,255,0.16);
        }}
        .pill-college {{
            background: rgba(217, 119, 6, 0.22);
            color: #FBBF24;
            border-color: rgba(251, 191, 36, 0.45);
        }}
        .pill-engine {{
            background: rgba(2, 132, 199, 0.25);
            color: #38BDF8;
            border-color: rgba(56, 189, 248, 0.45);
        }}
        .pill-latency {{
            background: rgba(16, 185, 129, 0.22);
            color: #34D399;
            border-color: rgba(52, 211, 153, 0.45);
        }}
        .pill-mode {{
            background: rgba(148, 163, 184, 0.18);
            color: #E2E8F0;
        }}

        /* KPI Telemetry Card */
        .csmss-kpi-card {{
            background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-top: 3px solid #0284C7;
            border-radius: 10px;
            padding: 12px 14px;
            min-height: 106px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
        }}
        .csmss-kpi-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .csmss-kpi-value {{
            font-size: 1.55rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.2;
        }}
        .csmss-kpi-delta {{
            display: inline-block;
            margin-top: 6px;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
        }}

        /* Severity & Status Badges */
        .sev-badge {{
            display: inline-block;
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .sev-Critical {{ background: rgba(239, 68, 68, 0.22); color: #FCA5A5; border: 1px solid #EF4444; }}
        .sev-High     {{ background: rgba(249, 115, 22, 0.22); color: #FDBA74; border: 1px solid #F97316; }}
        .sev-Medium   {{ background: rgba(245, 158, 11, 0.22); color: #FDE68A; border: 1px solid #F59E0B; }}
        .sev-Low      {{ background: rgba(16, 185, 129, 0.22); color: #6EE7B7; border: 1px solid #10B981; }}

        /* Section Sub-card */
        .csmss-section-card {{
            background: #1E293B;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_system_status_header(
    engine_mode: str,
    latency_ms: float,
    connection_mode: str,
    active_role_label: str,
) -> None:
    """Render the persistent top CSMSS Kanchanwadi Civic Monitor v2.0 status banner."""
    st.markdown(
        f"""
        <div class="csmss-top-header">
            <div class="csmss-title-row">
                <div>
                    <div class="csmss-main-title">
                        🛡️ {APP_TITLE}
                    </div>
                    <div class="csmss-sub-title">
                        🏛️ {INSTITUTION_NAME} • Active Role: <b>{active_role_label}</b>
                    </div>
                </div>
                <div class="csmss-pill-group">
                    <span class="csmss-pill pill-college">📍 CSMSS KANCHANWADI CAMPUS</span>
                    <span class="csmss-pill pill-engine">⚙️ {engine_mode}</span>
                    <span class="csmss-pill pill-latency">⚡ {latency_ms:.1f} ms Frame Latency</span>
                    <span class="csmss-pill pill-mode">🔗 {connection_mode}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    delta_text: str,
    accent_hex: str = "#0284C7",
    delta_bg_rgba: str = "rgba(16, 185, 129, 0.18)",
    delta_text_hex: str = "#34D399",
) -> None:
    """Render a styled KPI metric card with top accent bar and delta badge."""
    st.markdown(
        f"""
        <div class="csmss-kpi-card" style="border-top-color: {accent_hex};">
            <div class="csmss-kpi-label">{label}</div>
            <div class="csmss-kpi-value">{value}</div>
            <div class="csmss-kpi-delta" style="background: {delta_bg_rgba}; color: {delta_text_hex};">
                {delta_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_severity_badge(severity: str) -> str:
    """Return an inline HTML badge for `Low`, `Medium`, `High`, or `Critical`."""
    sev_clean = severity if severity in {"Low", "Medium", "High", "Critical"} else "Medium"
    return f'<span class="sev-badge sev-{sev_clean}">{sev_clean}</span>'