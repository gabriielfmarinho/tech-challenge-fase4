import warnings

import cv2

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import face_recognition


def detect_faces_with_haar(frame_bgr, scale_factor=1.1, min_neighbors=5, min_size=40):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    frontal = cv2.CascadeClassifier(frontal_path)
    profile = cv2.CascadeClassifier(profile_path)
    faces = []
    for classifier in (frontal, profile):
        detected = classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        for x, y, w, h in detected:
            faces.append((y, x + w, y + h, x))
    return faces


def filter_faces(frame_shape, face_boxes, min_size=40, min_ratio=0.6, max_ratio=1.6):
    height, width = frame_shape[:2]
    filtered = []
    for top, right, bottom, left in face_boxes:
        box_width = right - left
        box_height = bottom - top
        if box_width < min_size or box_height < min_size:
            continue
        if top < 0 or left < 0 or right > width or bottom > height:
            continue
        ratio = box_width / float(box_height) if box_height else 0.0
        if ratio < min_ratio or ratio > max_ratio:
            continue
        filtered.append((top, right, bottom, left))
    return filtered


def detect_faces(
    frame_bgr,
    model="hog",
    upsample=1,
    fallback="haar",
    haar_scale=1.1,
    haar_neighbors=5,
    min_size=40,
):
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_boxes = face_recognition.face_locations(
        rgb_frame, number_of_times_to_upsample=upsample, model=model
    )
    if not face_boxes and fallback == "haar":
        face_boxes = detect_faces_with_haar(
            frame_bgr,
            scale_factor=haar_scale,
            min_neighbors=haar_neighbors,
            min_size=min_size,
        )
    return filter_faces(frame_bgr.shape, face_boxes, min_size=min_size)


def draw_face_boxes(frame_bgr, face_boxes):
    for top, right, bottom, left in face_boxes:
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    return frame_bgr
