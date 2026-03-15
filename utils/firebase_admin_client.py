import base64
import json
import os

try:
    import firebase_admin
    from firebase_admin import auth, credentials
except Exception:
    firebase_admin = None
    auth = None
    credentials = None


def _init_firebase_admin():
    if firebase_admin is None or auth is None or credentials is None:
        return False, "firebase-admin package is not installed."

    if firebase_admin._apps:
        return True, None

    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    service_account_b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_BASE64")

    try:
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            return True, None

        if service_account_b64:
            decoded = base64.b64decode(service_account_b64).decode("utf-8")
            info = json.loads(decoded)
            cred = credentials.Certificate(info)
            firebase_admin.initialize_app(cred)
            return True, None
    except Exception as exc:
        return False, str(exc)

    return (
        False,
        "Firebase Admin not configured. Set FIREBASE_SERVICE_ACCOUNT_PATH or FIREBASE_SERVICE_ACCOUNT_BASE64.",
    )


def delete_firebase_user_by_email(email):
    ok, err = _init_firebase_admin()
    if not ok:
        return {"ok": False, "error": err}

    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return {"ok": False, "error": "email_required"}

    try:
        user = auth.get_user_by_email(clean_email)
        auth.delete_user(user.uid)
        return {"ok": True, "deleted_uid": user.uid}
    except auth.UserNotFoundError:
        return {"ok": True, "deleted_uid": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def delete_all_firebase_users(batch_size=1000):
    ok, err = _init_firebase_admin()
    if not ok:
        return {"ok": False, "error": err, "deleted_count": 0}

    deleted_count = 0
    page = auth.list_users()

    while page:
        uids = [user.uid for user in page.users]
        if uids:
            result = auth.delete_users(uids)
            deleted_count += int(getattr(result, "success_count", 0))
        page = page.get_next_page()

    return {"ok": True, "deleted_count": deleted_count}
