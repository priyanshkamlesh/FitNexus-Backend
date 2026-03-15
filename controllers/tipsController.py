import os
import json
from flask import request, jsonify

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _get_groq_config():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_TEXT_MODEL", "llama-3.1-8b-instant")
    return api_key, model_name


def api_generate_tips():
    api_key, model_name = _get_groq_config()

    if OpenAI is None:
        return jsonify({"error": "openai package is not installed on backend."}), 500

    if not api_key:
        return jsonify({"error": "Groq not configured. Set GROQ_API_KEY in backend/.env"}), 500

    data = request.get_json(silent=True) or {}
    activity = (data.get("activity") or "posture").strip()
    posture_data = data.get("posture_data") or {}
    score = posture_data.get("score")
    feedback_list = posture_data.get("feedback") or []

    if not isinstance(feedback_list, list):
        feedback_list = []

    prompt = f"""
You are an expert posture and movement coach.

The user has uploaded an image and posture analysis produced:

Activity: {activity}
Posture Score: {score}

Joint Feedback:
{json.dumps(feedback_list, indent=2)}

Write practical improvement tips.

Rules:
- Address user as "you"
- Start with short posture summary
- Give 4-8 bullet points
- Keep language simple
- Use markdown
""".strip()

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a concise posture coach."},
                {"role": "user", "content": prompt},
            ],
        )

        tips = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            if isinstance(content, str):
                tips = content.strip()

        if not tips:
            return jsonify({"error": "Groq returned empty response"}), 502

        return jsonify({"tips": tips}), 200

    except Exception as e:
        message = str(e)
        lowered = message.lower()
        status_code = getattr(e, "status_code", None)
        print("Groq tips error:", message)

        if status_code == 429 or "429" in lowered or "rate limit" in lowered or "quota" in lowered:
            return jsonify({"error": "Groq rate limit exceeded. Please retry shortly."}), 429
        if status_code == 401:
            return jsonify({"error": "Invalid GROQ_API_KEY."}), 401
        if status_code == 400:
            return jsonify({"error": f"Groq rejected request. Check model '{model_name}'."}), 400

        return jsonify({"error": "Failed to generate tips"}), 500
