from flask import request, jsonify
from datetime import datetime
from werkzeug.security import check_password_hash

from models.otpModel import (
    save_otp,
    get_latest_otp,
    mark_verified,
    increase_attempts,
)

from utils.otp import generate_otp
from utils.email import send_email


OTP_EXPIRY_MINUTES = 10
OTP_MAX_ATTEMPTS = 5


def send_otp():

    data = request.get_json(silent=True) or {}

    email = (data.get("email")).strip().lower()

    if not email:
        return jsonify({"error": "email required"}), 400

    otp = generate_otp()

    save_otp(email, otp, OTP_EXPIRY_MINUTES)

    subject = "Your verification code"

    body = f"Your OTP code is {otp}. Valid for {OTP_EXPIRY_MINUTES} minutes."

    send_email(email, subject, body)

    return jsonify({"ok": True}), 200


def verify_otp():

    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()

    if not email or not otp:
        return jsonify({"error": "email and otp required"}), 400

    row = get_latest_otp(email)

    if not row:
        return jsonify({"error": "no otp found"}), 404

    if row["verified"]:
        return jsonify({"error": "already verified"}), 400

    expires_at = datetime.fromisoformat(row["expires_at"])

    if datetime.utcnow() > expires_at:
        return jsonify({"error": "otp expired"}), 400

    attempts = row["attempts"] or 0

    if attempts >= OTP_MAX_ATTEMPTS:
        return jsonify({"error": "too many attempts"}), 403

    if check_password_hash(row["otp_hash"], otp):

        mark_verified(row["id"])

        return jsonify({"ok": True}), 200

    else:

        increase_attempts(row["id"])

        return jsonify({
            "error": "invalid otp",
            "attempts_left": OTP_MAX_ATTEMPTS - (attempts + 1)
        }), 400