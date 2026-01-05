import math

import cv2
import mediapipe as mp


def create_activity_state():
    pose = None
    if hasattr(mp, "solutions"):
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return {
        "pose": pose,
        "prev_landmarks": None,
        "prev_gray": None,
    }


def detect_activity(frame_bgr, state):
    if state["pose"]:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = state["pose"].process(rgb_frame)
        if not result.pose_landmarks:
            state["prev_landmarks"] = None
            return "unknown", 0.0

        landmarks = result.pose_landmarks.landmark
        movement_score = compute_movement(landmarks, state["prev_landmarks"])
        activity = classify_activity(landmarks, movement_score)
        state["prev_landmarks"] = landmarks
        return activity, movement_score
    return detect_activity_by_motion(frame_bgr, state)


def classify_activity(landmarks, movement):
    arm_raised = is_arm_raised(landmarks)
    if arm_raised and movement > 0.01:
        return "gesturing"
    if movement > 0.02:
        return "high_motion"
    if movement > 0.005:
        return "low_motion"
    return "idle"


def compute_movement(landmarks, prev_landmarks):
    if not prev_landmarks:
        return 0.0
    total = 0.0
    count = 0
    for current, previous in zip(landmarks, prev_landmarks):
        total += math.hypot(current.x - previous.x, current.y - previous.y)
        count += 1
    return total / count if count else 0.0


def is_arm_raised(landmarks):
    left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_raised = left_wrist.y < left_shoulder.y
    right_raised = right_wrist.y < right_shoulder.y
    return left_raised or right_raised


def detect_activity_by_motion(frame_bgr, state):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if state["prev_gray"] is None:
        state["prev_gray"] = gray
        return "unknown", 0.0
    diff = cv2.absdiff(state["prev_gray"], gray)
    motion_score = diff.mean() / 255.0
    state["prev_gray"] = gray
    if motion_score > 0.08:
        return "high_motion", motion_score
    if motion_score > 0.03:
        return "low_motion", motion_score
    return "idle", motion_score


def draw_activity(frame_bgr, activity):
    cv2.putText(
        frame_bgr,
        activity,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr
