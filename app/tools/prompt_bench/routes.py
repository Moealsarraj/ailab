"""Prompt Bench routes."""
from flask import Blueprint, render_template, request, jsonify
from .bench import run_bench, recommend_fixes

bp = Blueprint("prompt_bench", __name__, template_folder="templates")


@bp.route("/")
def index():
    return render_template("prompt_bench/index.html")


@bp.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(silent=True) or {}
    system_prompt = (body.get("system_prompt") or "").strip()
    test_cases    = body.get("test_cases") or []

    if not system_prompt:
        return jsonify({"error": "system_prompt is required"}), 400
    if not test_cases or not isinstance(test_cases, list):
        return jsonify({"error": "test_cases must be a non-empty list"}), 400
    if len(test_cases) > 10:
        return jsonify({"error": "Maximum 10 test cases per run"}), 400

    result = run_bench(system_prompt, test_cases)
    return jsonify(result)


@bp.route("/api/recommend-fixes", methods=["POST"])
def api_recommend_fixes():
    body        = request.get_json(silent=True) or {}
    system      = (body.get("system_prompt") or "").strip()
    run_results = body.get("run_results") or []

    if not system:
        return jsonify({"error": "system_prompt is required"}), 400
    if not run_results:
        return jsonify({"error": "run_results is required"}), 400

    result = recommend_fixes(system, run_results)
    return jsonify(result)
