from utils.db import users_collection, scores_collection
from werkzeug.security import generate_password_hash
from pymongo import ReturnDocument
from datetime import datetime, timezone


def create_user(email, password, username):

    password_hash = generate_password_hash(password)
    now = datetime.now(timezone.utc)

    users_collection.insert_one({
        "email": email.lower(),
        "password_hash": password_hash,
        "username": username,
        "provider": "local",
        "email_verified": False,
        "profile": {},
        "created_at": now,
        "updated_at": now,
    })


def find_user(email):
    if not email:
        return None

    user = users_collection.find_one({
        "email": email.lower()
    })

    return user


def upsert_user_profile(email, profile):
    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return None

    now = datetime.now(timezone.utc)
    username = str(profile.get("username") or "").strip()
    name = str(profile.get("name") or "").strip()

    clean_profile = {
        "name": name,
        "username": username,
        "height": profile.get("height"),
        "weight": profile.get("weight"),
        "age": profile.get("age"),
        "gender": str(profile.get("gender") or "").strip(),
        "purpose": str(profile.get("purpose") or "").strip(),
        "bmi": profile.get("bmi"),
    }

    user = users_collection.find_one_and_update(
        {"email": clean_email},
        {
            "$set": {
                "profile": clean_profile,
                "username": username or name,
                "updated_at": now,
            },
            "$setOnInsert": {
                "email": clean_email,
                "provider": "firebase",
                "email_verified": True,
                "created_at": now,
            },
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    return user


def delete_user_account(email):
    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return {"deleted_user_count": 0, "deleted_score_count": 0}

    user_result = users_collection.delete_one({"email": clean_email})
    score_result = scores_collection.delete_many({"email": clean_email})

    return {
        "deleted_user_count": user_result.deleted_count,
        "deleted_score_count": score_result.deleted_count,
    }


def delete_all_user_accounts():
    user_result = users_collection.delete_many({})
    score_result = scores_collection.delete_many({})

    return {
        "deleted_user_count": user_result.deleted_count,
        "deleted_score_count": score_result.deleted_count,
    }


def get_user_profile(email):
    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return None

    user = users_collection.find_one({"email": clean_email})
    if not user:
        return None

    profile = user.get("profile") or {}
    return {
        "email": clean_email,
        "username": profile.get("username") or user.get("username") or "",
        "age": profile.get("age") or "",
        "gender": profile.get("gender") or "Male",
        "profile": profile,
    }
