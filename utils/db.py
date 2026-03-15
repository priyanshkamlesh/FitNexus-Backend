from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv("keys.env")

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

# use database from URI
db = client["SageCoachDB"]

users_collection = db["users"]
otp_collection = db["otps"]
scores_collection = db["scores"]

try:
    client.admin.command("ping")
    print("✅ MongoDB connected successfully")
except Exception as e:
    print("❌ MongoDB connection failed:", e)