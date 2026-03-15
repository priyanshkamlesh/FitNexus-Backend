from utils.db import scores_collection
from bson import ObjectId
import json


# -----------------------------
# Insert score
# -----------------------------
def insert_score(email, score, meta):

    scores_collection.insert_one({
        "email": email,
        "score": float(score),
        "meta": meta
    })


# -----------------------------
# Get scores
# -----------------------------
def fetch_scores(email=None):

    if email:
        cursor = scores_collection.find({"email": email}).sort("_id", -1)
    else:
        cursor = scores_collection.find().sort("_id", -1).limit(100)

    rows = []

    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        rows.append(doc)

    return rows


# -----------------------------
# Get only score list (for charts)
# -----------------------------
def fetch_score_values():

    cursor = scores_collection.find({}, {"score": 1})

    return [doc["score"] for doc in cursor]


# -----------------------------
# Delete all scores
# -----------------------------
def delete_all_scores():

    scores_collection.delete_many({})