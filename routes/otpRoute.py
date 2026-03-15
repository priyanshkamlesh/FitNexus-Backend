from flask import Blueprint
from controllers.otpController import send_otp, verify_otp

otp_bp = Blueprint("otp", __name__)

otp_bp.route("/api/send_otp", methods=["POST"])(send_otp)
otp_bp.route("/api/verify_otp", methods=["POST"])(verify_otp)