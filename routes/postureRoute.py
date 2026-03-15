from flask import Blueprint
from controllers.postureController import api_analyze, api_classify_exercise_frame

posture_bp = Blueprint("posture", __name__)

posture_bp.route("/analyze", methods=["POST"])(api_analyze)
posture_bp.route("/classify_exercise_frame", methods=["POST"])(api_classify_exercise_frame)
