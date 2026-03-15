from flask import request, jsonify
from models.scoreModels import (
    insert_score,
    fetch_scores,
    fetch_score_values,
    delete_all_scores,
)


def api_save_score():

    data = request.get_json(silent=True) or {}

    score = data.get("score")
    email = (data.get("email") or "").strip().lower() or None
    meta = data.get("meta") or {}

    if score is None:
        return jsonify({"error": "score required"}), 400

    try:
        insert_score(email, score, meta)
    except Exception as e:
        print("save score error:", e)
        return jsonify({"error": "database error"}), 500

    return jsonify({"ok": True}), 201


def api_get_scores():

    email = (request.args.get("email") or "").strip().lower()

    rows = fetch_scores(email)

    return jsonify(rows), 200


def scores_alias_save():
    return api_save_score()


def scores_alias_get():

    scores = fetch_score_values()

    return jsonify(scores), 200


def scores_alias_delete_all():

    delete_all_scores()

    return jsonify({
        "ok": True,
        "message": "all scores deleted"
    }), 200