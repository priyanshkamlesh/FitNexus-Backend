from flask import jsonify

def register_error_handlers(app):

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error":"route_not_found"}),404