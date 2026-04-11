"""Arabic Bench routes."""
from flask import Blueprint, render_template, request, jsonify
from .bench import evaluate_arabic

bp = Blueprint("arabic_bench", __name__, template_folder="templates")


@bp.route("/")
def index():
    return render_template("arabic_bench/index.html")


@bp.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    body = request.get_json(silent=True) or {}
    ai_response = (body.get("ai_response") or "").strip()
    reference   = (body.get("reference") or "").strip()

    if not ai_response:
        return jsonify({"error": "Paste the AI response to evaluate"}), 400
    if not reference:
        return jsonify({"error": "Paste the reference answer to compare against"}), 400
    if len(ai_response) < 10 or len(reference) < 10:
        return jsonify({"error": "Both texts are too short to evaluate"}), 400

    result = evaluate_arabic(ai_response, reference)
    if not result:
        return jsonify({"error": "Evaluation failed — please try again"}), 502
    return jsonify(result)
