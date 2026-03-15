from flask import Blueprint
from controllers.scoreController import (
    api_save_score,
    api_get_scores,
    scores_alias_save,
    scores_alias_get,
    scores_alias_delete_all,
)

score_bp = Blueprint("scores", __name__)

score_bp.route("/api/scores", methods=["POST"])(api_save_score)
score_bp.route("/api/scores", methods=["GET"])(api_get_scores)

score_bp.route("/scores", methods=["POST"])(scores_alias_save)
score_bp.route("/scores", methods=["GET"])(scores_alias_get)
score_bp.route("/scores", methods=["DELETE"])(scores_alias_delete_all)