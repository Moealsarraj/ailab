"""Prompt Shield routes."""
from flask import Blueprint, render_template, request, jsonify
from .shield import audit_prompt

bp = Blueprint("prompt_shield", __name__, template_folder="templates")


@bp.route("/")
def index():
    return render_template("prompt_shield/index.html")


@bp.route("/api/audit", methods=["POST"])
def api_audit():
    body = request.get_json(silent=True) or {}
    system_prompt = (body.get("system_prompt") or "").strip()

    if not system_prompt:
        return jsonify({"error": "Paste a system prompt to audit"}), 400
    if len(system_prompt) < 20:
        return jsonify({"error": "System prompt too short to audit"}), 400
    if len(system_prompt) > 8000:
        return jsonify({"error": "System prompt too long — keep it under 8000 characters"}), 400

    try:
        result = audit_prompt(system_prompt)
    except Exception:
        return jsonify({"error": "AI audit failed — please try again"}), 502
    if not result:
        return jsonify({"error": "AI audit failed — please try again"}), 502
    return jsonify(result)
