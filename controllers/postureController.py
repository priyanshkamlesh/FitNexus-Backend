import numpy as np
import cv2
from flask import request, jsonify
from utils.posture import analyze_posture_image
from utils.exercise_classifier import (
    classify_exercise_image,
    decode_base64_image,
)


def api_analyze():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:

        file_bytes = np.frombuffer(file.read(), np.uint8)

        if file_bytes.size == 0:
            return jsonify({"error": "Uploaded file is empty"}), 400

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        expected_exercise = request.form.get("expected_exercise", "").strip().lower() or None
        expected_phase = request.form.get("expected_phase", "").strip().lower() or None
        result = analyze_posture_image(
            img,
            expected_exercise=expected_exercise,
            expected_phase=expected_phase,
        )

        return jsonify({
            "ok": True,
            "posture_score": result["score"],
            "feedback": result["feedback"],
            "joint_status": result.get("joint_status", {}),
            "annotated_image_base64": result["annotated_b64"],
            "corrected_skeleton_image_base64": result["ideal_b64"]
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print("Posture analyze error:", e)
        return jsonify({"error": "Failed to analyze posture"}), 500


def api_classify_exercise_frame():
    try:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            image_base64 = payload.get("image_base64")
            if not image_base64:
                return jsonify({"error": "Missing image_base64"}), 400
            image = decode_base64_image(image_base64)
        elif "file" in request.files:
            file = request.files["file"]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            if file_bytes.size == 0:
                return jsonify({"error": "Uploaded frame is empty"}), 400
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"error": "Invalid image format"}), 400
        else:
            return jsonify({"error": "No frame uploaded"}), 400

        result = classify_exercise_image(image)
        return jsonify({"ok": True, **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("Exercise classify error:", e)
        return jsonify({"error": "Failed to classify exercise"}), 500
