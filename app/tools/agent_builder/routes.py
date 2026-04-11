"""Agent Builder routes."""
from flask import Blueprint, render_template, request, jsonify
from .builder import build_agent

bp = Blueprint("agent_builder", __name__, template_folder="templates")


@bp.route("/")
def index():
    return render_template("agent_builder/index.html")


@bp.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(silent=True) or {}
    description = (body.get("description") or "").strip()

    if not description:
        return jsonify({"error": "Describe your agent first"}), 400
    if len(description) < 20:
        return jsonify({"error": "Description too short — be more specific about what the agent should do"}), 400
    if len(description) > 1000:
        return jsonify({"error": "Description too long — keep it under 1000 characters"}), 400

    try:
        result = build_agent(description)
    except Exception as e:
        return jsonify({"error": "AI failed to generate — please try again"}), 502
    if not result:
        return jsonify({"error": "AI failed to generate — please try again"}), 502
    return jsonify(result)
