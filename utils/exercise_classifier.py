import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except ImportError:
    mp_pose = None


REFERENCE_PATH = Path(__file__).resolve().parent.parent / "data" / "exercise_reference.json"

LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

ANGLE_FEATURES = [
    "right_elbow_right_shoulder_right_hip",
    "left_elbow_left_shoulder_left_hip",
    "right_knee_mid_hip_left_knee",
    "right_hip_right_knee_right_ankle",
    "left_hip_left_knee_left_ankle",
    "right_wrist_right_elbow_right_shoulder",
    "left_wrist_left_elbow_left_shoulder",
]

DISTANCE_FEATURES = [
    "left_shoulder_left_wrist",
    "right_shoulder_right_wrist",
    "left_hip_left_ankle",
    "right_hip_right_ankle",
    "left_hip_left_wrist",
    "right_hip_right_wrist",
    "left_shoulder_left_ankle",
    "right_shoulder_right_ankle",
    "left_hip_right_wrist",
    "right_hip_left_wrist",
    "left_elbow_right_elbow",
    "left_knee_right_knee",
    "left_wrist_right_wrist",
    "left_ankle_right_ankle",
    "left_hip_avg_left_wrist_left_ankle",
    "right_hip_avg_right_wrist_right_ankle",
]


def _load_reference():
    if not REFERENCE_PATH.exists():
        raise RuntimeError(
            f"Exercise reference file not found at {REFERENCE_PATH}. Build it first."
        )
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


REFERENCE = None


def get_reference():
    global REFERENCE
    if REFERENCE is None:
        REFERENCE = _load_reference()
    return REFERENCE


def decode_base64_image(image_base64: str) -> np.ndarray:
    payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    data = base64.b64decode(payload)
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data.")
    return image


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _distance(a, b):
    return float(np.linalg.norm(a - b))


def _avg_point(a, b):
    return (a + b) / 2.0


def _vector_angle_from_horizontal(a, b):
    vector = b - a
    return abs(math.degrees(math.atan2(vector[1], vector[0])))


def _extract_pose_data(image_bgr: np.ndarray):
    if mp_pose is None:
        raise RuntimeError(
            "mediapipe not installed. Run: pip install mediapipe opencv-python numpy"
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks or not results.pose_world_landmarks:
        raise ValueError("Could not detect a full body pose in the frame.")

    image_landmarks = results.pose_landmarks.landmark
    world_landmarks = results.pose_world_landmarks.landmark

    coords = {}
    for name in LANDMARK_NAMES:
        enum_value = mp_pose.PoseLandmark[name]
        landmark = world_landmarks[enum_value.value]
        coords[name] = np.array(
            [landmark.x * 100.0, landmark.y * 100.0, landmark.z * 100.0],
            dtype=float,
        )

    image_coords = {}
    for name in LANDMARK_NAMES:
        enum_value = mp_pose.PoseLandmark[name]
        landmark = image_landmarks[enum_value.value]
        image_coords[name] = np.array([landmark.x, landmark.y], dtype=float)

    mid_hip = _avg_point(coords["LEFT_HIP"], coords["RIGHT_HIP"])

    features = {
        "right_elbow_right_shoulder_right_hip": _angle(
            coords["RIGHT_ELBOW"], coords["RIGHT_SHOULDER"], coords["RIGHT_HIP"]
        ),
        "left_elbow_left_shoulder_left_hip": _angle(
            coords["LEFT_ELBOW"], coords["LEFT_SHOULDER"], coords["LEFT_HIP"]
        ),
        "right_knee_mid_hip_left_knee": _angle(
            coords["RIGHT_KNEE"], mid_hip, coords["LEFT_KNEE"]
        ),
        "right_hip_right_knee_right_ankle": _angle(
            coords["RIGHT_HIP"], coords["RIGHT_KNEE"], coords["RIGHT_ANKLE"]
        ),
        "left_hip_left_knee_left_ankle": _angle(
            coords["LEFT_HIP"], coords["LEFT_KNEE"], coords["LEFT_ANKLE"]
        ),
        "right_wrist_right_elbow_right_shoulder": _angle(
            coords["RIGHT_WRIST"], coords["RIGHT_ELBOW"], coords["RIGHT_SHOULDER"]
        ),
        "left_wrist_left_elbow_left_shoulder": _angle(
            coords["LEFT_WRIST"], coords["LEFT_ELBOW"], coords["LEFT_SHOULDER"]
        ),
        "left_shoulder_left_wrist": _distance(coords["LEFT_SHOULDER"], coords["LEFT_WRIST"]),
        "right_shoulder_right_wrist": _distance(coords["RIGHT_SHOULDER"], coords["RIGHT_WRIST"]),
        "left_hip_left_ankle": _distance(coords["LEFT_HIP"], coords["LEFT_ANKLE"]),
        "right_hip_right_ankle": _distance(coords["RIGHT_HIP"], coords["RIGHT_ANKLE"]),
        "left_hip_left_wrist": _distance(coords["LEFT_HIP"], coords["LEFT_WRIST"]),
        "right_hip_right_wrist": _distance(coords["RIGHT_HIP"], coords["RIGHT_WRIST"]),
        "left_shoulder_left_ankle": _distance(coords["LEFT_SHOULDER"], coords["LEFT_ANKLE"]),
        "right_shoulder_right_ankle": _distance(coords["RIGHT_SHOULDER"], coords["RIGHT_ANKLE"]),
        "left_hip_right_wrist": _distance(coords["LEFT_HIP"], coords["RIGHT_WRIST"]),
        "right_hip_left_wrist": _distance(coords["RIGHT_HIP"], coords["LEFT_WRIST"]),
        "left_elbow_right_elbow": _distance(coords["LEFT_ELBOW"], coords["RIGHT_ELBOW"]),
        "left_knee_right_knee": _distance(coords["LEFT_KNEE"], coords["RIGHT_KNEE"]),
        "left_wrist_right_wrist": _distance(coords["LEFT_WRIST"], coords["RIGHT_WRIST"]),
        "left_ankle_right_ankle": _distance(coords["LEFT_ANKLE"], coords["RIGHT_ANKLE"]),
        "left_hip_avg_left_wrist_left_ankle": _distance(
            coords["LEFT_HIP"], _avg_point(coords["LEFT_WRIST"], coords["LEFT_ANKLE"])
        ),
        "right_hip_avg_right_wrist_right_ankle": _distance(
            coords["RIGHT_HIP"], _avg_point(coords["RIGHT_WRIST"], coords["RIGHT_ANKLE"])
        ),
    }

    mid_shoulder_2d = _avg_point(image_coords["LEFT_SHOULDER"], image_coords["RIGHT_SHOULDER"])
    mid_hip_2d = _avg_point(image_coords["LEFT_HIP"], image_coords["RIGHT_HIP"])
    mid_ankle_2d = _avg_point(image_coords["LEFT_ANKLE"], image_coords["RIGHT_ANKLE"])
    avg_wrist_y = float((image_coords["LEFT_WRIST"][1] + image_coords["RIGHT_WRIST"][1]) / 2.0)
    avg_shoulder_y = float((image_coords["LEFT_SHOULDER"][1] + image_coords["RIGHT_SHOULDER"][1]) / 2.0)

    posture_metrics = {
        "body_tilt_angle": _vector_angle_from_horizontal(mid_shoulder_2d, mid_ankle_2d),
        "torso_tilt_angle": _vector_angle_from_horizontal(mid_shoulder_2d, mid_hip_2d),
        "wrists_above_shoulders": avg_wrist_y < avg_shoulder_y - 0.04,
        "wrists_below_shoulders": avg_wrist_y > avg_shoulder_y + 0.04,
        "avg_knee_angle": float(
            (
                features["right_hip_right_knee_right_ankle"] +
                features["left_hip_left_knee_left_ankle"]
            ) / 2.0
        ),
    }

    return {
        "features": features,
        "posture_metrics": posture_metrics,
    }


def extract_pose_features(image_bgr: np.ndarray):
    return _extract_pose_data(image_bgr)["features"]


def _normalize_features(features, reference):
    normalized = []
    for feature_name in reference["feature_names"]:
        mean = float(reference["feature_means"][feature_name])
        std = float(reference["feature_stds"][feature_name]) or 1.0
        normalized.append((float(features[feature_name]) - mean) / std)
    return np.array(normalized, dtype=float)


def classify_exercise_image(image_bgr: np.ndarray):
    reference = get_reference()
    pose_data = _extract_pose_data(image_bgr)
    features = pose_data["features"]
    posture_metrics = pose_data["posture_metrics"]
    vector = _normalize_features(features, reference)

    pose_scores = []
    for pose_label, centroid in reference["pose_centroids"].items():
        distance = float(np.linalg.norm(vector - np.array(centroid, dtype=float)))
        pose_scores.append((pose_label, distance))
    pose_scores.sort(key=lambda item: item[1])

    exercise_scores = []
    for exercise_name, centroid in reference["exercise_centroids"].items():
        distance = float(np.linalg.norm(vector - np.array(centroid, dtype=float)))
        exercise_scores.append((exercise_name, distance))
    exercise_scores.sort(key=lambda item: item[1])

    best_pose, best_pose_distance = pose_scores[0]
    best_exercise, best_exercise_distance = exercise_scores[0]
    runner_up_distance = exercise_scores[1][1] if len(exercise_scores) > 1 else best_exercise_distance + 1.0

    heuristic_override = None
    body_tilt_angle = posture_metrics["body_tilt_angle"]
    avg_knee_angle = posture_metrics["avg_knee_angle"]

    if (
        body_tilt_angle <= 65
        and avg_knee_angle <= 135
    ):
        heuristic_override = "situp"
    elif body_tilt_angle <= 35 and posture_metrics["wrists_below_shoulders"] and avg_knee_angle >= 145:
        heuristic_override = "pushups"
    elif body_tilt_angle >= 55 and posture_metrics["wrists_above_shoulders"]:
        heuristic_override = "pullups"

    if heuristic_override in {"pushups", "pullups", "situp"}:
        best_exercise = heuristic_override
        matching_pose = next(
            (
                pose_label
                for pose_label, _ in pose_scores
                if pose_label.startswith(f"{heuristic_override}_")
            ),
            best_pose,
        )
        best_pose = matching_pose

    confidence = max(
        0.0,
        min(1.0, 1.0 - (best_exercise_distance / (runner_up_distance + 1e-6)) * 0.7),
    )
    if heuristic_override == best_exercise:
        confidence = max(confidence, 0.68)

    phase = best_pose.rsplit("_", 1)[1] if "_" in best_pose else ""

    return {
        "exercise": best_exercise,
        "pose_label": best_pose,
        "phase": phase,
        "confidence": round(confidence, 3),
        "exercise_distance": round(best_exercise_distance, 3),
        "body_tilt_angle": round(body_tilt_angle, 1),
        "avg_knee_angle": round(avg_knee_angle, 1),
        "heuristic_override": heuristic_override,
    }
