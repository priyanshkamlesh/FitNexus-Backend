from flask import Blueprint
from controllers.authController import (
    register,
    login,
    save_profile,
    get_profile,
    user_exists,
    delete_account,
    dev_delete_account,
    dev_delete_all_accounts,
)

auth_bp = Blueprint("auth",__name__)

auth_bp.route("/api/register",methods=["POST"])(register)
auth_bp.route("/api/login",methods=["POST"])(login)
auth_bp.route("/api/user/profile", methods=["POST"])(save_profile)
auth_bp.route("/api/user/profile", methods=["GET"])(get_profile)
auth_bp.route("/api/user/exists", methods=["POST"])(user_exists)
auth_bp.route("/api/user/delete-account", methods=["POST"])(delete_account)
auth_bp.route("/api/dev/accounts/delete-one", methods=["POST"])(dev_delete_account)
auth_bp.route("/api/dev/accounts/delete-all", methods=["DELETE"])(dev_delete_all_accounts)
