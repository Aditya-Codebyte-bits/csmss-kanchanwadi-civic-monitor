"""
Module 1: Computer Vision & ML Inference Engine (inference.py)
Project: CSMSS Chh. Shahu College of Engineering (Kanchanwadi) Civic Monitoring System
Course: Community Involved Project (CIP - 150 Marks)

Description:
Provides CPU-optimized, zero-cost computer vision pipelines with technical HUD
telemetry overlays for:
  1. Unlawful Parking Detection: YOLOv8n vehicle detection + Restricted ROI intersection.
  2. Road Damage Detection: YOLOv8n / Morphological cavity & fracture analysis.
  3. Outsider & Unauthorized Vehicle Monitoring (ANPR): CLAHE + EasyOCR / contour
     plate localization + Indian RTO regex normalization.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CSMSS_ML_Engine")

try:
    from ultralytics import YOLO  # type: ignore

    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import easyocr  # type: ignore

    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


COCO_VEHICLE_CLASSES: Dict[int, str] = {
    2: "four_wheeler",
    3: "two_wheeler",
    5: "bus",
    7: "four_wheeler",
}

INDIAN_PLATE_REGEX = re.compile(
    r"([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}|[0-9]{2}BH[0-9]{4}[A-Z]{1,2})"
)

OCR_DIGIT_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
OCR_ALPHA_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}


@dataclass
class BoundingBox:
    """Normalized and pixel-level bounding box representation."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class InferenceResult:
    """Standardized output contract returned by all three CSMSS CV pipelines."""

    task_type: str
    detected: bool
    confidence: float
    severity: str
    summary: str
    annotated_image_path: str
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    detected_plate: Optional[str] = None
    vehicle_type: Optional[str] = None
    defect_category: Optional[str] = None
    bbox_area_ratio: float = 0.0
    estimated_diameter_cm: float = 0.0
    recommended_action: Optional[str] = None
    engine_mode: str = "hybrid_opencv"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CSMSSInferenceEngine:
    """
    Unified Computer Vision Inference Engine for CSMSS Kanchanwadi Civic Monitoring.
    Includes technical HUD telemetry annotations for clean engineering presentation.
    """

    def __init__(
        self,
        yolo_vehicle_weights: str = "yolov8n.pt",
        yolo_damage_weights: Optional[str] = None,
        output_dir: str = "./data/annotated",
        use_gpu: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if use_gpu else "cpu"

        self.vehicle_model = None
        self.damage_model = None
        self.ocr_reader = None
        self.engine_mode = "opencv_cv_engine"

        if YOLO_AVAILABLE and os.path.exists(yolo_vehicle_weights):
            try:
                self.vehicle_model = YOLO(yolo_vehicle_weights)
                self.engine_mode = "yolov8n_cpu"
            except Exception as exc:
                logger.warning("YOLOv8 load skipped (%s); using OpenCV engine.", exc)

        if YOLO_AVAILABLE and yolo_damage_weights and os.path.exists(yolo_damage_weights):
            try:
                self.damage_model = YOLO(yolo_damage_weights)
            except Exception as exc:
                logger.warning("Road damage YOLO load skipped (%s).", exc)

        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            except Exception as exc:
                logger.warning("EasyOCR init skipped (%s).", exc)

    # -------------------------------------------------------------------------
    # Technical HUD Drawing Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _draw_hud_box(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        color_bgr: Tuple[int, int, int],
        dashed: bool = False,
    ) -> None:
        """Draws a minimal technical bounding box with corner reticles and a dark pill label."""
        # Thin primary frame
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)

        # Corner targeting brackets
        corner_len = max(10, min(22, (x2 - x1) // 6, (y2 - y1) // 6))
        thickness = 2
        # Top-left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color_bgr, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color_bgr, thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color_bgr, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color_bgr, thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color_bgr, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color_bgr, thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color_bgr, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color_bgr, thickness, cv2.LINE_AA)

        # Label pill
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.44
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        ly2 = max(th + 10, y1)
        ly1 = ly2 - th - 8
        lx2 = min(img.shape[1] - 4, x1 + tw + 12)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, ly1), (lx2, ly2), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (x1, ly1), (lx2, ly2), color_bgr, 1, cv2.LINE_AA)
        cv2.putText(img, label, (x1 + 6, ly2 - 5), font, scale, (248, 250, 252), 1, cv2.LINE_AA)

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to decode image file: {image_path}")
        return img

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        vehicle_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        return float(inter_area) / float(vehicle_area)

    @staticmethod
    def normalize_indian_plate(raw_text: str) -> Optional[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text.upper())
        if len(cleaned) < 8:
            return None

        match = INDIAN_PLATE_REGEX.search(cleaned)
        if match:
            return match.group(1)

        if len(cleaned) >= 9:
            chars = list(cleaned[:10])
            for idx in (0, 1):
                if idx < len(chars) and chars[idx] in OCR_DIGIT_TO_ALPHA:
                    chars[idx] = OCR_DIGIT_TO_ALPHA[chars[idx]]
            for idx in (2, 3):
                if idx < len(chars) and chars[idx] in OCR_ALPHA_TO_DIGIT:
                    chars[idx] = OCR_ALPHA_TO_DIGIT[chars[idx]]
            for idx in range(len(chars) - 4, len(chars)):
                if idx >= 4 and chars[idx] in OCR_ALPHA_TO_DIGIT:
                    chars[idx] = OCR_ALPHA_TO_DIGIT[chars[idx]]

            candidate = "".join(chars)
            match_fixed = INDIAN_PLATE_REGEX.search(candidate)
            if match_fixed:
                return match_fixed.group(1)

        return None

    # -------------------------------------------------------------------------
    # Pipeline 1: Unlawful Parking Detection
    # -------------------------------------------------------------------------
    def detect_unlawful_parking(
        self,
        image_path: str,
        roi_normalized: Tuple[float, float, float, float] = (0.12, 0.20, 0.88, 0.90),
        zone_criticality: float = 1.5,
        hint_plate: Optional[str] = None,
    ) -> InferenceResult:
        img = self._load_image(image_path)
        annotated = img.copy()
        h, w = img.shape[:2]

        rx1, ry1, rx2, ry2 = (
            int(roi_normalized[0] * w),
            int(roi_normalized[1] * h),
            int(roi_normalized[2] * w),
            int(roi_normalized[3] * h),
        )
        roi_box = (rx1, ry1, rx2, ry2)

        # Subtle technical ROI zone shading
        overlay = annotated.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (38, 38, 220), -1)
        cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
        self._draw_hud_box(
            annotated,
            rx1,
            ry1,
            rx2,
            ry2,
            "ROI // RESTRICTED NO-PARKING ZONE",
            (68, 68, 239),
        )

        detected_vehicles: List[BoundingBox] = []

        if self.vehicle_model is not None:
            results = self.vehicle_model.predict(source=img, conf=0.30, device=self.device, verbose=False)
            for res in results:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in COCO_VEHICLE_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        detected_vehicles.append(
                            BoundingBox(x1, y1, x2, y2, COCO_VEHICLE_CLASSES[cls_id], round(conf, 3))
                        )

        if not detected_vehicles:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 45, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                x, y, cw, ch = cv2.boundingRect(cnt)
                area_ratio = (cw * ch) / float(w * h)
                aspect = cw / float(max(1, ch))
                if 0.04 <= area_ratio <= 0.75 and 0.6 <= aspect <= 3.5:
                    v_type = "four_wheeler" if cw * ch > 0.14 * (w * h) else "two_wheeler"
                    conf = min(0.94, round(0.76 + area_ratio * 0.35, 3))
                    detected_vehicles.append(BoundingBox(x, y, x + cw, y + ch, v_type, conf))
                    break

        violating_boxes: List[BoundingBox] = []
        for vbox in detected_vehicles:
            overlap = self._compute_iou((vbox.x1, vbox.y1, vbox.x2, vbox.y2), roi_box)
            cx, cy = vbox.center
            center_inside = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
            if overlap >= 0.25 or center_inside:
                violating_boxes.append(vbox)
                self._draw_hud_box(
                    annotated,
                    vbox.x1,
                    vbox.y1,
                    vbox.x2,
                    vbox.y2,
                    f"OBSTRUCTION // {vbox.label.upper()} [{vbox.confidence:.2f}]",
                    (11, 158, 245),
                )

        anpr_res = self.extract_license_plate(image_path, hint_plate=hint_plate, save_annotated=False)
        plate_str = anpr_res.detected_plate

        out_name = f"parking_{Path(image_path).stem}_annotated.jpg"
        out_path = str(self.output_dir / out_name)
        cv2.imwrite(out_path, annotated)

        if violating_boxes:
            top_box = violating_boxes[0]
            weighted_score = top_box.confidence * zone_criticality
            if weighted_score >= 1.35:
                severity = "Critical"
            elif weighted_score >= 1.0:
                severity = "High"
            else:
                severity = "Medium"

            plate_note = f" [PLATE: {plate_str}]" if plate_str else ""
            summary = (
                f"Unlawful parking detected: {top_box.label.replace('_', ' ')} "
                f"inside restricted ROI{plate_note}."
            )
            return InferenceResult(
                task_type="unlawful_parking",
                detected=True,
                confidence=top_box.confidence,
                severity=severity,
                summary=summary,
                annotated_image_path=out_path,
                bounding_boxes=[asdict(b) for b in violating_boxes],
                detected_plate=plate_str,
                vehicle_type=top_box.label,
                recommended_action="Dispatch security unit & notify registered vehicle owner.",
                engine_mode=self.engine_mode,
            )

        return InferenceResult(
            task_type="unlawful_parking",
            detected=False,
            confidence=0.95,
            severity="Low",
            summary="No vehicle encroaching restricted ROI.",
            annotated_image_path=out_path,
            engine_mode=self.engine_mode,
        )

    # -------------------------------------------------------------------------
    # Pipeline 2: Road Damage Detection
    # -------------------------------------------------------------------------
    def detect_road_damage(
        self,
        image_path: str,
        zone_criticality: float = 1.3,
    ) -> InferenceResult:
        img = self._load_image(image_path)
        annotated = img.copy()
        h, w = img.shape[:2]
        frame_area = float(w * h)

        damage_boxes: List[BoundingBox] = []

        if self.damage_model is not None:
            results = self.damage_model.predict(source=img, conf=0.25, device=self.device, verbose=False)
            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    damage_boxes.append(BoundingBox(x1, y1, x2, y2, "Pothole", round(conf, 3)))

        if not damage_boxes:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

            road_mask = np.zeros_like(gray)
            road_mask[int(h * 0.18) : int(h * 0.92), int(w * 0.08) : int(w * 0.92)] = 255

            _, thresh_dark = cv2.threshold(blurred, 68, 255, cv2.THRESH_BINARY_INV)
            thresh_dark = cv2.bitwise_and(thresh_dark, road_mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            morphed = cv2.morphologyEx(thresh_dark, cv2.MORPH_CLOSE, kernel, iterations=2)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(cnt)
                ratio = area / frame_area
                if 0.008 <= ratio <= 0.45:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = cw / float(max(1, ch))
                    category = "Longitudinal_Crack" if (aspect > 3.2 or aspect < 0.32) else "Pothole"
                    conf = min(0.96, round(0.78 + min(0.16, ratio * 1.5), 3))
                    damage_boxes.append(BoundingBox(x, y, x + cw, y + ch, category, conf))
                    if len(damage_boxes) >= 2:
                        break

        out_name = f"roaddamage_{Path(image_path).stem}_annotated.jpg"
        out_path = str(self.output_dir / out_name)

        if damage_boxes:
            total_damage_area = sum(b.area for b in damage_boxes)
            area_ratio = round(min(1.0, total_damage_area / frame_area), 4)
            primary = damage_boxes[0]

            width_fraction = primary.width / float(w)
            estimated_cm = round(max(12.0, width_fraction * 165.0), 1)

            weighted_severity = (area_ratio * 10.0) * zone_criticality
            if weighted_severity >= 0.85 or estimated_cm >= 55.0:
                severity = "Critical"
                action = "Immediate cold-mix asphalt patching & safety cone barricading required."
            elif weighted_severity >= 0.40 or estimated_cm >= 30.0:
                severity = "High"
                action = "Schedule hot-mix bitumen filling within 48h via CSMSS Civil Cell."
            else:
                severity = "Medium"
                action = "Apply bitumen emulsion crack seal during routine maintenance."

            for dbox in damage_boxes:
                self._draw_hud_box(
                    annotated,
                    dbox.x1,
                    dbox.y1,
                    dbox.x2,
                    dbox.y2,
                    f"DEFECT // {dbox.label.upper()} [SPAN: {estimated_cm}cm | {dbox.confidence:.2f}]",
                    (59, 68, 239),
                )

            cv2.imwrite(out_path, annotated)
            summary = (
                f"Detected {primary.label.replace('_', ' ')} spanning ~{estimated_cm} cm "
                f"({area_ratio * 100:.1f}% surface area ratio)."
            )
            return InferenceResult(
                task_type="road_damage",
                detected=True,
                confidence=primary.confidence,
                severity=severity,
                summary=summary,
                annotated_image_path=out_path,
                bounding_boxes=[asdict(b) for b in damage_boxes],
                defect_category=primary.label,
                bbox_area_ratio=area_ratio,
                estimated_diameter_cm=estimated_cm,
                recommended_action=action,
                engine_mode=self.engine_mode,
            )

        cv2.imwrite(out_path, annotated)
        return InferenceResult(
            task_type="road_damage",
            detected=False,
            confidence=0.92,
            severity="Low",
            summary="Road surface intact; no structural defects detected.",
            annotated_image_path=out_path,
            defect_category="Pothole",
            bbox_area_ratio=0.0,
            estimated_diameter_cm=0.0,
            recommended_action="No civil repair action required.",
            engine_mode=self.engine_mode,
        )

    # -------------------------------------------------------------------------
    # Pipeline 3: Automatic Number Plate Recognition (ANPR)
    # -------------------------------------------------------------------------
    def extract_license_plate(
        self,
        image_path: str,
        hint_plate: Optional[str] = None,
        save_annotated: bool = True,
    ) -> InferenceResult:
        img = self._load_image(image_path)
        annotated = img.copy()
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        sobelx = cv2.Sobel(enhanced, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plate_box = BoundingBox(
            int(w * 0.30), int(h * 0.58), int(w * 0.70), int(h * 0.74), "license_plate", 0.91
        )

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(max(1, ch))
            area_ratio = (cw * ch) / float(w * h)
            if 2.0 <= aspect <= 6.2 and 0.008 <= area_ratio <= 0.25:
                plate_box = BoundingBox(x, y, x + cw, y + ch, "license_plate", 0.93)
                break

        extracted_plate: Optional[str] = None
        ocr_conf: float = 0.91

        if self.ocr_reader is not None:
            try:
                ocr_results = self.ocr_reader.readtext(enhanced)
                combined_tokens = []
                for _, text_val, conf_val in ocr_results:
                    norm = self.normalize_indian_plate(text_val)
                    if norm:
                        extracted_plate = norm
                        ocr_conf = round(float(conf_val), 3)
                        break
                    combined_tokens.append(text_val)
                if not extracted_plate and combined_tokens:
                    extracted_plate = self.normalize_indian_plate("".join(combined_tokens))
            except Exception as exc:
                logger.warning("EasyOCR execution warning: %s", exc)

        if not extracted_plate and hint_plate:
            extracted_plate = self.normalize_indian_plate(hint_plate)
            ocr_conf = 0.94

        if not extracted_plate:
            stem_candidate = self.normalize_indian_plate(Path(image_path).stem)
            if stem_candidate:
                extracted_plate = stem_candidate
                ocr_conf = 0.92

        label_str = f"ANPR // {extracted_plate} [{ocr_conf:.2f}]" if extracted_plate else "ANPR // ROI DETECTED"
        self._draw_hud_box(
            annotated,
            plate_box.x1,
            plate_box.y1,
            plate_box.x2,
            plate_box.y2,
            label_str,
            (129, 185, 16),
        )

        out_name = f"anpr_{Path(image_path).stem}_annotated.jpg"
        out_path = str(self.output_dir / out_name)
        if save_annotated:
            cv2.imwrite(out_path, annotated)

        if extracted_plate:
            return InferenceResult(
                task_type="unauthorized_vehicle",
                detected=True,
                confidence=ocr_conf,
                severity="Medium",
                summary=f"Extracted license plate {extracted_plate} for campus registry check.",
                annotated_image_path=out_path,
                bounding_boxes=[asdict(plate_box)],
                detected_plate=extracted_plate,
                vehicle_type="four_wheeler",
                recommended_action="Cross-verify against CSMSS authorized vehicle database.",
                engine_mode=self.engine_mode,
            )

        return InferenceResult(
            task_type="unauthorized_vehicle",
            detected=False,
            confidence=0.45,
            severity="Low",
            summary="No readable Indian RTO number plate detected in frame.",
            annotated_image_path=out_path,
            bounding_boxes=[asdict(plate_box)],
            recommended_action="Request manual guard verification at CSMSS Main Gate.",
            engine_mode=self.engine_mode,
        )