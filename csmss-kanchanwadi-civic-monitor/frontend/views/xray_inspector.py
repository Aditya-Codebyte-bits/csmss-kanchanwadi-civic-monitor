"""
Tab 2 — 4-Stage CV X-Ray Inspector (`frontend/views/xray_inspector.py`).
CSMSS Kanchanwadi Civic Monitor v2.0 — CSMSS Chh. Shahu College of Engineering.

Features:
1. Select any of the 6 high-realism 1920x1080 campus test scenes OR upload a custom
   image/photo from a phone or laptop.
2. Display the 4-Stage Visual Pipeline Grid side-by-side:
   - `Stage 1: Raw Input`
   - `Stage 2: CLAHE/HSV/Edge Mask`
   - `Stage 3: ROI Polygon & Footprint Geometry`
   - `Stage 4: Tactical HUD Output`
3. Display extracted ANPR license plate crop image, Indian RTO regex match status,
   vehicle registry lookup result, and expandable raw JSON telemetry payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from config.settings import CAMPUS_ZONES
from cv_engine.anpr_reader import validate_and_normalize_indian_plate
from demo_suite.synthetic_generator import DEMO_FRAMES_DIR, SCENE_SPECS, generate_all_demo_frames
from frontend.styles import format_privacy_contact


def _ensure_demo_frames_exist() -> Dict[str, Path]:
    """Ensure all 6 procedural 1920x1080 demo frames exist on disk."""
    missing = any(not (DEMO_FRAMES_DIR / fname).exists() for fname in SCENE_SPECS)
    if missing:
        return generate_all_demo_frames(DEMO_FRAMES_DIR)
    return {fname: DEMO_FRAMES_DIR / fname for fname in SCENE_SPECS}


def render_xray_inspector_tab(
    pipeline: Any,
    vehicles_registry: List[Dict[str, Any]],
    cv_params: Dict[str, Any],
    active_role_label: str,
    on_log_incident: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> None:
    """Render Tab 2: 4-Stage Explainable Computer Vision X-Ray Inspector."""
    frame_paths = _ensure_demo_frames_exist()

    st.subheader("🔬 4-Stage Explainable Computer Vision X-Ray Inspector")
    st.caption(
        "Inspect every intermediate computer vision stage in real time — from raw camera feed to "
        "CLAHE/Canny/Black-Hat surface filtering, bitwise ROI polygon intersection, and tactical HUD synthesis."
    )

    # ------------------------------------------------------------------
    # 1. Scene Selector OR Custom Phone/Camera Image Upload
    # ------------------------------------------------------------------
    sel_col1, sel_col2, sel_col3 = st.columns([1.6, 1.0, 1.2])
    scene_options = list(SCENE_SPECS.keys())
    scene_labels = {
        fname: f"{fname} — {SCENE_SPECS[fname]['title']} ({SCENE_SPECS[fname]['zone_id']})"
        for fname in scene_options
    }

    with sel_col1:
        selected_scene_file = st.selectbox(
            "Select Preset CSMSS Campus Surveillance Scene (1920×1080)",
            options=scene_options,
            format_func=lambda k: scene_labels[k],
            key="xray_scene_selector",
        )
    spec = SCENE_SPECS[selected_scene_file]

    with sel_col2:
        zone_override = st.selectbox(
            "Camera Zone Calibration Context",
            options=["ZONE-01", "ZONE-02", "ZONE-03", "ZONE-04"],
            index=["ZONE-01", "ZONE-02", "ZONE-03", "ZONE-04"].index(spec["zone_id"]),
            format_func=lambda zid: f"{zid}: {CAMPUS_ZONES[zid]['zone_name']}",
            key="xray_zone_override",
        )

    with sel_col3:
        uploaded_file = st.file_uploader(
            "Or Upload Custom Campus / Phone Photo (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="xray_custom_upload",
        )

    # Load BGR image from upload or selected preset scene
    plate_hint: Optional[str] = spec.get("primary_plate")
    if uploaded_file is not None:
        raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        decoded_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            st.error("Could not decode uploaded image file.")
            return
        frame_bgr = decoded_bgr
        source_caption = f"Custom Upload: `{uploaded_file.name}` ({frame_bgr.shape[1]}×{frame_bgr.shape[0]})"
        plate_hint = None
    else:
        target_path = frame_paths[selected_scene_file]
        frame_bgr = cv2.imread(str(target_path))
        if frame_bgr is None:
            generate_all_demo_frames(DEMO_FRAMES_DIR)
            frame_bgr = cv2.imread(str(target_path))
        source_caption = f"Preset Scene: `{selected_scene_file}` — {spec['description']}"

    st.info(f"📷 **Active Frame Source:** {source_caption}")

    # Build whitelist set from current vehicle registry
    whitelisted_plates = [
        v["plate_number"] for v in vehicles_registry if v.get("is_whitelisted")
    ]

    # Run the 4-Stage CV Pipeline with live sidebar tuning parameters
    analysis = pipeline.analyze_frame(
        frame_bgr=frame_bgr,
        zone_id=zone_override,
        conf_threshold=float(cv_params.get("conf_threshold", 0.45)),
        roi_overlap_threshold=float(cv_params.get("roi_overlap_threshold", 0.30)),
        clahe_clip_limit=float(cv_params.get("clahe_clip_limit", 2.5)),
        whitelisted_plates=whitelisted_plates,
        plate_hint=plate_hint,
    )

    xray = analysis["xray_frames"]
    lat = analysis["latency_ms"]

    # Optional: respect sidebar toggles for ROI polygon / contour visibility in Stage 3
    stage3_img = xray["stage3_geometry"]
    if not cv_params.get("show_roi_polygons", True) and not cv_params.get("show_intermediate_contours", True):
        stage3_img = xray["stage1_raw"]

    # ------------------------------------------------------------------
    # 2. 4-Stage Visual Pipeline Grid (2×2 Side-by-Side Proof)
    # ------------------------------------------------------------------
    st.markdown("### 🖼️ 4-Stage Computer Vision Pipeline Proof")
    g_row1_col1, g_row1_col2 = st.columns(2)
    with g_row1_col1:
        st.markdown("**Stage 1: Raw Input (`stage1_raw`)**")
        st.image(
            xray["stage1_raw"],
            caption=f"Original Input Frame + Camera Zone Banner ({lat['preprocess_ms']:.1f} ms)",
            use_container_width=True,
        )
    with g_row1_col2:
        st.markdown("**Stage 2: CLAHE / HSV / Edge Mask (`stage2_preprocessed`)**")
        st.image(
            xray["stage2_preprocessed"],
            caption=(
                f"CLAHE (clip={cv_params.get('clahe_clip_limit', 2.5):.1f}) + Black-Hat Darkness + "
                f"Canny Edge & Turbo Anomaly Heatmap"
            ),
            use_container_width=True,
        )

    g_row2_col1, g_row2_col2 = st.columns(2)
    with g_row2_col1:
        st.markdown("**Stage 3: ROI Polygon & Footprint Geometry (`stage3_geometry`)**")
        st.image(
            stage3_img,
            caption=(
                f"Bitwise cv2.fillPoly ROI Mask + Vehicle Footprint Intersection "
                f"(Threshold >= {float(cv_params.get('roi_overlap_threshold', 0.30)) * 100:.0f}%) + Pothole Contour Mesh"
            ),
            use_container_width=True,
        )
    with g_row2_col2:
        st.markdown("**Stage 4: Tactical HUD Output (`stage4_hud`)**")
        st.image(
            xray["stage4_hud"],
            caption=(
                f"Final Tactical HUD • Corner-Bracket Detections, ANPR Callouts, Severity Index Bar "
                f"& GPS Watermark (Total: {lat['total_ms']:.1f} ms)"
            ),
            use_container_width=True,
        )

    st.markdown("<hr style='border-color: rgba(148,163,184,0.18); margin: 14px 0;'/>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. ANPR License Plate Crop, Regex Status & Vehicle Registry Lookup
    # ------------------------------------------------------------------
    st.markdown("### 🔍 ANPR License Plate Extraction & Campus Registry Verification")
    eval_vehicles = analysis["parking"]["evaluated_vehicles"]
    plate_crops = analysis["parking"]["plate_crops_rgb"]
    reg_lookup = {
        "".join(ch for ch in v["plate_number"].upper() if ch.isalnum()): v
        for v in vehicles_registry
    }

    if eval_vehicles:
        for idx, veh in enumerate(eval_vehicles):
            a_col1, a_col2, a_col3 = st.columns([0.9, 1.3, 1.4])
            plate_compact = veh.get("plate_number", "MH20EE4501")
            plate_fmt = veh.get("formatted_plate", plate_compact)
            _, _, regex_ok = validate_and_normalize_indian_plate(plate_fmt)
            reg_entry = reg_lookup.get(plate_compact)

            with a_col1:
                st.markdown(f"**Vehicle #{idx + 1} Cropped Plate ROI**")
                if idx < len(plate_crops) and plate_crops[idx] is not None:
                    st.image(
                        plate_crops[idx],
                        caption=f"Bilateral + CLAHE Enhanced Crop ({plate_fmt})",
                        use_container_width=True,
                    )
            with a_col2:
                st.markdown("**Indian RTO Regex & Polygon ROI Telemetry**")
                st.markdown(
                    f"- **Normalized Plate:** `{plate_fmt}` (`{plate_compact}`)\n"
                    f"- **Indian RTO Regex Match (`MH 20`):** {'✅ Valid Indian RTO Format' if regex_ok else '⚠️ Non-Standard'}\n"
                    f"- **OCR Confidence:** `{float(veh.get('ocr_confidence', 0.90)) * 100:.1f}%` ({veh.get('ocr_backend')})\n"
                    f"- **ROI Overlap Ratio:** `{float(veh.get('overlap_ratio', 0.0)) * 100:.1f}%` "
                    f"(Threshold: `{float(cv_params.get('roi_overlap_threshold', 0.30)) * 100:.0f}%`)\n"
                    f"- **Classification:** **{veh.get('incident_type')}** (`{veh.get('severity')}`)"
                )
            with a_col3:
                st.markdown("**CSMSS Campus Vehicle Registry Lookup**")
                if reg_entry:
                    masked_phone = format_privacy_contact(
                        reg_entry.get("contact_no", ""), active_role_label
                    )
                    wl_badge = (
                        "✅ Whitelisted Campus Pass"
                        if reg_entry.get("is_whitelisted")
                        else "🚫 Suspended / Not Whitelisted"
                    )
                    st.markdown(
                        f"- **Registry Status:** {wl_badge}\n"
                        f"- **Registered Owner:** `{reg_entry.get('owner_name')}`\n"
                        f"- **Campus Role & Type:** `{reg_entry.get('role')}` • `{reg_entry.get('vehicle_type')}`\n"
                        f"- **Contact Number:** `{masked_phone}`"
                    )
                else:
                    st.warning(
                        f"🚨 **Unregistered External Vehicle (`{plate_fmt}`)**: Plate not found in CSMSS Kanchanwadi `registered_vehicles` database."
                    )
    else:
        st.info("No vehicles detected in this frame (Road-Surface-Only Scene).")

    # Action Button to commit detected candidate incidents into SQLite / Deduplicator
    candidates = analysis.get("candidate_incidents", [])
    if candidates and on_log_incident is not None:
        if st.button(
            f"📥 Commit {len(candidates)} Detected Hazard(s) from This Frame to Temporal Deduplicator",
            type="primary",
            key="xray_commit_btn",
        ):
            for cand in candidates:
                res = on_log_incident(cand)
                if res.get("deduplicated"):
                    st.toast(
                        f"Deduplicated Incident #{res['incident_id']} (repeat_count -> {res['repeat_count']})",
                        icon="♻️",
                    )
                else:
                    st.toast(
                        f"Logged New Incident #{res['incident_id']} ({cand['incident_type']})",
                        icon="🚨",
                    )

    # ------------------------------------------------------------------
    # 4. Expandable Raw JSON Telemetry Payload
    # ------------------------------------------------------------------
    with st.expander("🧾 View Raw Explainable CV Telemetry Payload (JSON)", expanded=False):
        serializable_payload = {
            "engine_mode": analysis["engine_mode"],
            "zone_id": analysis["zone_id"],
            "parameters": analysis["parameters"],
            "latency_ms": analysis["latency_ms"],
            "parking_violations": analysis["parking"]["violations"],
            "road_damage_summary": analysis["road_damage"]["summary"],
            "road_damage_cavities": analysis["road_damage"]["cavities"],
            "candidate_incidents": analysis["candidate_incidents"],
        }
        st.json(serializable_payload)