from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
from utils.db import otp_collection
from bson import ObjectId


def save_otp(email, otp, expiry_minutes=10):

    otp_hash = generate_password_hash(otp)

    expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)

    otp_collection.insert_one({
        "email": email.lower(),
        "otp_hash": otp_hash,
        "expires_at": expires_at,
        "attempts": 0,
        "verified": False,
        "created_at": datetime.utcnow()
    })


def get_latest_otp(email):

    return otp_collection.find_one(
        {"email": email.lower()},
        sort=[("created_at", -1)]
    )


def mark_verified(otp_id):

    otp_collection.update_one(
        {"_id": ObjectId(otp_id)},
        {"$set": {"verified": True}}
    )


def increase_attempts(otp_id):

    otp_collection.update_one(
        {"_id": ObjectId(otp_id)},
        {"$inc": {"attempts": 1}}
    )