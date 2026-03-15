from flask import Blueprint
from controllers.tipsController import api_generate_tips

tips_bp = Blueprint("tips", __name__)

tips_bp.route("/generate_tips", methods=["POST"])(api_generate_tips)