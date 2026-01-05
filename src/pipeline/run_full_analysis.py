import json
import os
from collections import Counter

import cv2

from config.settings import (
    DEFAULT_ANALYSIS_METADATA_FILE,
    DEFAULT_ANALYSIS_OUTPUT_DIR,
    DEFAULT_ANALYSIS_OUTPUT_VIDEO,
    DEFAULT_FACE_FALLBACK,
    DEFAULT_FACE_MODEL,
    DEFAULT_FACE_PADDING,
    DEFAULT_HAAR_NEIGHBORS,
    DEFAULT_HAAR_SCALE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_UPSAMPLE,
)
from modules.emotion_analysis_module import analyze_emotions, draw_emotions
from modules.face_recognition_module import detect_faces, draw_face_boxes
from modules.activity_detection_module import ActivityDetector, draw_activity
from pipeline.run_face_recognition import scale_boxes


def build_output_paths(output_dir, output_video, metadata_file):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, output_video)
    metadata_path = os.path.join(output_dir, metadata_file)
    return video_path, metadata_path


def create_writer(capture, output_path):
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer, fps


def run_pipeline(
    input_path,
    output_dir,
    output_video,
    metadata_file,
    frame_step=1,
    max_frames=None,
    resize_width=None,
    face_model=DEFAULT_FACE_MODEL,
    upsample=DEFAULT_UPSAMPLE,
    face_fallback=DEFAULT_FACE_FALLBACK,
    haar_scale=DEFAULT_HAAR_SCALE,
    haar_neighbors=DEFAULT_HAAR_NEIGHBORS,
    summary_only=True,
    min_face_size=DEFAULT_MIN_FACE_SIZE,
    face_padding=DEFAULT_FACE_PADDING,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    output_video_path, metadata_path = build_output_paths(
        output_dir, output_video, metadata_file
    )
    writer, fps = create_writer(capture, output_video_path)

    frame_index = 0
    processed_frames = 0
    activity_detector = ActivityDetector()
    emotion_counts = Counter()
    activity_counts = Counter()
    faces_detected = 0
    anomaly_count = 0
    motion_window = []
    with open(metadata_path, "w", encoding="utf-8") as metadata_handle:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_step > 1 and frame_index % frame_step != 0:
                writer.write(frame)
                frame_index += 1
                continue

            frame_for_detection = frame
            scale_x = 1.0
            scale_y = 1.0
            if resize_width:
                height, width = frame.shape[:2]
                resize_height = int(height * (resize_width / float(width)))
                frame_for_detection = cv2.resize(frame, (resize_width, resize_height))
                scale_x = width / float(resize_width)
                scale_y = height / float(resize_height)

            face_boxes = detect_faces(
                frame_for_detection,
                model=face_model,
                upsample=upsample,
                fallback=face_fallback,
                haar_scale=haar_scale,
                haar_neighbors=haar_neighbors,
                min_size=min_face_size,
            )
            if scale_x != 1.0 or scale_y != 1.0:
                face_boxes = scale_boxes(face_boxes, scale_x, scale_y)

            emotions = analyze_emotions(frame, face_boxes, face_padding=face_padding)
            emotion_counts.update(emotions)
            faces_detected += len(emotions)
            activity, motion_score = activity_detector.detect(frame_for_detection)
            activity_counts.update([activity])
            is_anomaly = False
            if motion_score:
                motion_window.append(motion_score)
                if len(motion_window) > 30:
                    motion_window.pop(0)
                avg_motion = sum(motion_window) / len(motion_window)
                if motion_score > avg_motion * 2.5 and motion_score > 0.02:
                    is_anomaly = True
                    anomaly_count += 1
            annotated_frame = draw_face_boxes(frame, face_boxes)
            annotated_frame = draw_emotions(annotated_frame, face_boxes, emotions)
            annotated_frame = draw_activity(annotated_frame, activity)
            writer.write(annotated_frame)

            if not summary_only:
                timestamp = frame_index / fps if fps else 0.0
                record = {
                    "frame_index": int(frame_index),
                    "timestamp": float(timestamp),
                    "face_count": int(len(face_boxes)),
                    "boxes": [list(map(int, box)) for box in face_boxes],
                    "emotions": emotions,
                    "activity": activity,
                    "motion_score": float(motion_score),
                    "is_anomaly": is_anomaly,
                }
                metadata_handle.write(json.dumps(record) + "\n")

            frame_index += 1
            processed_frames += 1
            if max_frames and processed_frames >= max_frames:
                break

        summary = {
            "frames_processed": processed_frames,
            "faces_detected": faces_detected,
            "anomalies_detected": anomaly_count,
            "activities": dict(activity_counts),
            "emotions": dict(emotion_counts),
            "top_activities": [
                {"label": label, "count": count}
                for label, count in activity_counts.most_common(3)
            ],
            "top_emotions": [
                {"label": label, "count": count}
                for label, count in emotion_counts.most_common(3)
            ],
        }
        metadata_handle.write(json.dumps({"summary": summary}) + "\n")

    capture.release()
    writer.release()


def run_full_analysis(
    input_path,
    output_dir=None,
    output_video=None,
    metadata_file=None,
    frame_step=1,
    max_frames=None,
    resize_width=None,
    face_model=DEFAULT_FACE_MODEL,
    upsample=DEFAULT_UPSAMPLE,
    face_fallback=DEFAULT_FACE_FALLBACK,
    haar_scale=DEFAULT_HAAR_SCALE,
    haar_neighbors=DEFAULT_HAAR_NEIGHBORS,
    summary_only=True,
    min_face_size=DEFAULT_MIN_FACE_SIZE,
    face_padding=DEFAULT_FACE_PADDING,
):
    resolved_output_dir = output_dir or DEFAULT_ANALYSIS_OUTPUT_DIR
    resolved_output_video = output_video or DEFAULT_ANALYSIS_OUTPUT_VIDEO
    resolved_metadata_file = metadata_file or DEFAULT_ANALYSIS_METADATA_FILE
    resolved_face_model = face_model or DEFAULT_FACE_MODEL
    resolved_upsample = DEFAULT_UPSAMPLE if upsample is None else upsample
    resolved_face_fallback = face_fallback or DEFAULT_FACE_FALLBACK
    resolved_haar_scale = DEFAULT_HAAR_SCALE if haar_scale is None else haar_scale
    resolved_haar_neighbors = (
        DEFAULT_HAAR_NEIGHBORS if haar_neighbors is None else haar_neighbors
    )
    resolved_min_face_size = (
        DEFAULT_MIN_FACE_SIZE if min_face_size is None else min_face_size
    )
    resolved_face_padding = (
        DEFAULT_FACE_PADDING if face_padding is None else face_padding
    )
    run_pipeline(
        input_path,
        resolved_output_dir,
        resolved_output_video,
        resolved_metadata_file,
        frame_step=frame_step,
        max_frames=max_frames,
        resize_width=resize_width,
        face_model=resolved_face_model,
        upsample=resolved_upsample,
        face_fallback=resolved_face_fallback,
        haar_scale=resolved_haar_scale,
        haar_neighbors=resolved_haar_neighbors,
        summary_only=summary_only,
        min_face_size=resolved_min_face_size,
        face_padding=resolved_face_padding,
    )
