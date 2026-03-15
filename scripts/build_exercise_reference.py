import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


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

FEATURE_NAMES = ANGLE_FEATURES + DISTANCE_FEATURES


def parse_csv_from_zip(zf, name):
    with zf.open(name) as file_obj:
        return list(csv.DictReader((line.decode("utf-8") for line in file_obj)))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values, value_mean):
    if not values:
        return 1.0
    variance = sum((value - value_mean) ** 2 for value in values) / len(values)
    return variance ** 0.5 or 1.0


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 build_exercise_reference.py <archive.zip> <output.json>")
        raise SystemExit(1)

    archive_path = Path(sys.argv[1]).expanduser().resolve()
    output_path = Path(sys.argv[2]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as zf:
        labels_rows = parse_csv_from_zip(zf, "labels.csv")
        angles_rows = parse_csv_from_zip(zf, "angles.csv")
        distance_rows = parse_csv_from_zip(zf, "3d_distances.csv")

    labels_by_id = {row["pose_id"]: row["pose"] for row in labels_rows}
    angles_by_id = {row["pose_id"]: row for row in angles_rows}
    distances_by_id = {row["pose_id"]: row for row in distance_rows}

    feature_rows = []
    for pose_id, pose_label in labels_by_id.items():
        if pose_id not in angles_by_id or pose_id not in distances_by_id:
            continue

        row = []
        for feature_name in FEATURE_NAMES:
            source = angles_by_id[pose_id] if feature_name in ANGLE_FEATURES else distances_by_id[pose_id]
            row.append(float(source[feature_name]))

        feature_rows.append((pose_label, row))

    per_feature_values = defaultdict(list)
    for _, row in feature_rows:
        for feature_name, value in zip(FEATURE_NAMES, row):
            per_feature_values[feature_name].append(value)

    means = {feature_name: mean(values) for feature_name, values in per_feature_values.items()}
    stds = {
        feature_name: stddev(values, means[feature_name])
        for feature_name, values in per_feature_values.items()
    }

    grouped = defaultdict(list)
    grouped_by_exercise = defaultdict(list)

    for pose_label, row in feature_rows:
        normalized = [
            (value - means[feature_name]) / stds[feature_name]
            for feature_name, value in zip(FEATURE_NAMES, row)
        ]
        grouped[pose_label].append(normalized)
        exercise_name = pose_label.rsplit("_", 1)[0]
        grouped_by_exercise[exercise_name].append(normalized)

    pose_centroids = {
        pose_label: [
            mean([row[idx] for row in rows])
            for idx in range(len(FEATURE_NAMES))
        ]
        for pose_label, rows in grouped.items()
    }
    exercise_centroids = {
        exercise_name: [
            mean([row[idx] for row in rows])
            for idx in range(len(FEATURE_NAMES))
        ]
        for exercise_name, rows in grouped_by_exercise.items()
    }

    output = {
        "feature_names": FEATURE_NAMES,
        "feature_means": means,
        "feature_stds": stds,
        "pose_centroids": pose_centroids,
        "exercise_centroids": exercise_centroids,
    }

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
