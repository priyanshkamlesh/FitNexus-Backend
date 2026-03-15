from flask import request, jsonify
from models.userModels import (
    create_user,
    find_user,
    upsert_user_profile,
    delete_user_account,
    delete_all_user_accounts,
    get_user_profile,
)
from werkzeug.security import check_password_hash
from utils.firebase_admin_client import (
    delete_firebase_user_by_email,
    delete_all_firebase_users,
)

def register():

    data = request.json

    email = data.get("email")
    password = data.get("password")
    username = data.get("username")

    if find_user(email):
        return jsonify({"error":"user_exists"}),400

    create_user(email,password,username)

    return jsonify({"ok":True})

# Login User

def login():

    data = request.json

    email = data.get("email")
    password = data.get("password")

    user = find_user(email)

    if not user:
        return jsonify({"error":"invalid"}),401

    if not check_password_hash(user["password_hash"],password):
        return jsonify({"error":"invalid"}),401

    return jsonify({"ok":True})


def save_profile():
    data = request.json or {}

    email = data.get("email")
    profile = data.get("profile") or {}

    required = ["name", "username", "height", "weight", "age", "gender", "purpose"]
    missing = [field for field in required if profile.get(field) in (None, "")]
    if not email:
        return jsonify({"error": "email_required"}), 400
    if missing:
        return jsonify({"error": "missing_fields", "fields": missing}), 400

    saved_user = upsert_user_profile(email, profile)
    if not saved_user:
        return jsonify({"error": "save_failed"}), 500

    return jsonify({
        "ok": True,
        "message": "profile_saved",
        "email": saved_user.get("email"),
        "profile": saved_user.get("profile", {}),
    })


def get_profile():
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "email_required"}), 400

    profile = get_user_profile(email)
    if not profile:
        return jsonify({"ok": True, "exists": False, "profile": None}), 200

    return jsonify({"ok": True, "exists": True, "profile": profile}), 200


def user_exists():
    data = request.json or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "email_required"}), 400

    user = find_user(email)
    return jsonify({"ok": True, "exists": bool(user)})


def delete_account():
    data = request.json or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "email_required"}), 400

    result = delete_user_account(email)
    return jsonify({
        "ok": True,
        "deleted_user_count": result["deleted_user_count"],
        "deleted_score_count": result["deleted_score_count"],
    })


def dev_delete_account():
    data = request.json or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "email_required"}), 400

    mongo_result = delete_user_account(email)
    firebase_result = delete_firebase_user_by_email(email)

    return jsonify({
        "ok": True,
        "email": str(email).strip().lower(),
        "mongo": mongo_result,
        "firebase": firebase_result,
    })


def dev_delete_all_accounts():
    mongo_result = delete_all_user_accounts()
    firebase_result = delete_all_firebase_users()

    return jsonify({
        "ok": True,
        "mongo": mongo_result,
        "firebase": firebase_result,
    })
