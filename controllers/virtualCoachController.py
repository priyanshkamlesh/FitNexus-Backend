import os
from flask import request, jsonify

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_INSTRUCTION = """You are a specialized health, fitness, and nutrition expert.
Your goal is to provide accurate and helpful information.
Your HIGHEST PRIORITY is to provide answers in a list of bullet points.
Use bullet points (- text) or numbered points (1. text) for ALL answers.
Keep the points short and precise.

You can answer questions about:
- Diseases and health conditions
- Food, nutrition, and calories
- Sports and fitness exercises
- Creating diet plans and exercise routines

When a user asks for a plan (like a diet or exercise plan), ask only ONE follow-up question at a time.
If you provide medical advice, include a brief disclaimer to consult a professional.
""".strip()


def _get_groq_config():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_TEXT_MODEL", "llama-3.1-8b-instant")
    return api_key, model_name


def api_virtual_coach_chat():
    api_key, model_name = _get_groq_config()

    if OpenAI is None:
        return jsonify({
            "error": "openai package is not installed on backend."
        }), 500

    if not api_key:
        return jsonify({
            "error": "Groq not configured on backend. Set GROQ_API_KEY in backend environment."
        }), 500

    data = request.get_json(silent=True) or {}
    history = data.get("history") or []

    if not isinstance(history, list):
        return jsonify({"error": "Invalid payload: history must be a list."}), 400

    if not history:
        return jsonify({"error": "Invalid payload: history is empty."}), 400

    normalized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        sender = msg.get("sender")
        text = (msg.get("text") or "").strip()
        if sender not in ("user", "bot") or not text:
            continue
        normalized.append({"sender": sender, "text": text})

    if not normalized:
        return jsonify({"error": "No valid messages provided."}), 400

    if normalized[-1]["sender"] != "user":
        return jsonify({"error": "Last message must be from user."}), 400

    user_message = normalized[-1]["text"]
    previous_messages = normalized[:-1][-20:]

    try:
        # Groq exposes an OpenAI-compatible Chat Completions API.
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]

        for msg in previous_messages:
            role = "user" if msg["sender"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["text"]})

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        reply = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            if isinstance(content, str):
                reply = content.strip()
            elif isinstance(content, list):
                # Handle multimodal/structured content blocks if returned.
                chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
                reply = "\n".join(chunk for chunk in chunks if chunk).strip()

        if not reply:
            return jsonify({"error": "Groq returned empty response."}), 502

        return jsonify({"reply": reply}), 200
    except Exception as e:
        message = str(e)
        status_code = getattr(e, "status_code", None)
        print("Virtual coach Groq error:", message)

        lowered = message.lower()
        if status_code == 429 or "429" in lowered or "quota" in lowered or "rate limit" in lowered:
            return jsonify({"error": "Groq rate limit exceeded. Please retry shortly."}), 429
        if status_code == 401:
            return jsonify({"error": "Invalid GROQ_API_KEY."}), 401
        if status_code == 400:
            return jsonify({"error": f"Groq rejected the request. Check model '{model_name}' and payload format."}), 400

        return jsonify({"error": f"Failed to generate coach response: {message}"}), 500
