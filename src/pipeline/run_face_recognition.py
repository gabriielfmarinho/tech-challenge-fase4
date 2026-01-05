import json
import os

import cv2

from config.settings import (
    DEFAULT_FACE_MODEL,
    DEFAULT_METADATA_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_UPSAMPLE,
)
from modules.face_recognition_module import detect_faces, draw_face_boxes


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


def scale_boxes(face_boxes, scale_x, scale_y):
    scaled = []
    for top, right, bottom, left in face_boxes:
        scaled.append(
            (
                int(top * scale_y),
                int(right * scale_x),
                int(bottom * scale_y),
                int(left * scale_x),
            )
        )
    return scaled


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

            face_boxes = detect_faces(frame_for_detection, model=face_model, upsample=upsample)
            if scale_x != 1.0 or scale_y != 1.0:
                face_boxes = scale_boxes(face_boxes, scale_x, scale_y)

            annotated_frame = draw_face_boxes(frame, face_boxes)
            writer.write(annotated_frame)

            timestamp = frame_index / fps if fps else 0.0
            record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "face_count": len(face_boxes),
                "boxes": [list(box) for box in face_boxes],
            }
            metadata_handle.write(json.dumps(record) + "\n")
            frame_index += 1
            processed_frames += 1
            if max_frames and processed_frames >= max_frames:
                break

    capture.release()
    writer.release()


def run_face_recognition(
    input_path,
    output_dir=None,
    output_video=None,
    metadata_file=None,
    frame_step=1,
    max_frames=None,
    resize_width=None,
    face_model=DEFAULT_FACE_MODEL,
    upsample=DEFAULT_UPSAMPLE,
):
    resolved_output_dir = output_dir or DEFAULT_OUTPUT_DIR
    resolved_output_video = output_video or DEFAULT_OUTPUT_VIDEO
    resolved_metadata_file = metadata_file or DEFAULT_METADATA_FILE
    resolved_face_model = face_model or DEFAULT_FACE_MODEL
    resolved_upsample = DEFAULT_UPSAMPLE if upsample is None else upsample
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
    )
