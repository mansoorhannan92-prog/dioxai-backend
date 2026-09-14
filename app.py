"""
DioxAI backend proxy.

Keeps the Gemini API key on the server instead of exposing it in the
browser. The widget calls POST /api/chat; this forwards to Google's
Interactions API and keeps per-session conversation state server-side
using previous_interaction_id.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-real-key"
    export ALLOWED_ORIGIN="https://dioxai.online"   # your site's origin
    python app.py

The widget's BACKEND_URL should point at wherever you deploy this
(e.g. https://api.dioxai.online/api/chat) instead of Google directly.
"""

import os
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)

MODEL = "gemini-flash-latest"
SYSTEM_PROMPT = (
    "You are DioxAI, a helpful assistant embedded on dioxai.online. "
    "Answer clearly and concisely."
)

# In-memory session store: session_id -> last interaction id.
# Fine for a single-process demo; swap for Redis or a DB in production
# (multiple server processes/instances won't share this dict).
_sessions = {}


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not message:
        return jsonify({"error": "Message is required."}), 400
    if len(message) > 4000:
        return jsonify({"error": "Message is too long."}), 400

    previous_id = _sessions.get(session_id)

    try:
        kwargs = {
            "model": MODEL,
            "input": message,
            "generation_config": {"thinking_level": "low"},
        }
        if previous_id:
            kwargs["previous_interaction_id"] = previous_id
        else:
            kwargs["system_instruction"] = SYSTEM_PROMPT

        interaction = client.interactions.create(**kwargs)
        _sessions[session_id] = interaction.id

        reply = getattr(interaction, "output_text", None)
        if not reply:
            reply = interaction.outputs[-1].text

    except Exception as exc:  # noqa: BLE001 - surface a clean message to the widget
        return jsonify({"error": f"Assistant error: {exc}"}), 502

    return jsonify({"reply": reply, "session_id": session_id})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
          
