from flask import Blueprint
from controllers.virtualCoachController import api_virtual_coach_chat

virtual_coach_bp = Blueprint("virtual_coach", __name__)

virtual_coach_bp.route("/virtual_coach/chat", methods=["POST"])(api_virtual_coach_chat)
