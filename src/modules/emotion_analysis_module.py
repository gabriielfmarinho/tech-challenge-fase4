import os
import warnings

import cv2

from config.settings import DEFAULT_DEEPFACE_HOME

os.environ.setdefault("DEEPFACE_HOME", DEFAULT_DEEPFACE_HOME)
os.makedirs(DEFAULT_DEEPFACE_HOME, exist_ok=True)
from deepface import DeepFace

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


def expand_box(box, frame_shape, padding=0.15):
    top, right, bottom, left = box
    height, width = frame_shape[:2]
    box_height = bottom - top
    box_width = right - left
    pad_y = int(box_height * padding)
    pad_x = int(box_width * padding)
    new_top = max(0, top - pad_y)
    new_bottom = min(height, bottom + pad_y)
    new_left = max(0, left - pad_x)
    new_right = min(width, right + pad_x)
    return new_top, new_right, new_bottom, new_left


def analyze_emotions(frame_bgr, face_boxes, face_padding=0.15):
    emotions = []
    for top, right, bottom, left in face_boxes:
        box = (top, right, bottom, left)
        padded = expand_box(box, frame_bgr.shape, padding=face_padding)
        p_top, p_right, p_bottom, p_left = padded
        face_region = frame_bgr[p_top:p_bottom, p_left:p_right]
        if face_region.size == 0:
            emotions.append("unknown")
            continue
        try:
            result = DeepFace.analyze(
                face_region,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
            )
            if isinstance(result, list) and result:
                emotions.append(result[0].get("dominant_emotion", "unknown"))
            else:
                emotions.append(result.get("dominant_emotion", "unknown"))
        except Exception:
            emotions.append("unknown")
    return emotions


def draw_emotions(frame_bgr, face_boxes, emotions):
    for (top, right, bottom, left), emotion in zip(face_boxes, emotions):
        cv2.putText(
            frame_bgr,
            emotion,
            (left, max(top - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame_bgr
