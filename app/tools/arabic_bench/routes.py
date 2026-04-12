"""Arabic Bench routes."""
from flask import Blueprint, render_template, request, jsonify
from .bench import evaluate_arabic
from .benchmark import run_benchmark
from .dataset import CATEGORIES, DATASET, DATASET_BY_ID, DATASET_BY_CATEGORY
from app.core.ai import get_available_providers

bp = Blueprint("arabic_bench", __name__, template_folder="templates")


@bp.route("/")
def index():
    return render_template("arabic_bench/index.html",
                           categories=CATEGORIES, dataset=DATASET)


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


@bp.route("/api/dataset")
def api_dataset():
    """Return the full test dataset with categories."""
    category = request.args.get("category", "").strip()
    if category and category in DATASET_BY_CATEGORY:
        cases = DATASET_BY_CATEGORY[category]
    else:
        cases = DATASET
    return jsonify({"categories": CATEGORIES, "cases": cases})


@bp.route("/api/dataset/<case_id>")
def api_dataset_case(case_id):
    """Return a single test case by ID."""
    case = DATASET_BY_ID.get(case_id)
    if not case:
        return jsonify({"error": "Test case not found"}), 404
    return jsonify(case)


@bp.route("/api/providers")
def api_providers():
    """Return list of available AI providers for benchmarking."""
    return jsonify({"providers": get_available_providers()})


@bp.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """Run a multi-model benchmark on an Arabic prompt.

    Body: { "prompt": str, "reference": str, "providers": [str] (optional) }
    If providers is omitted, all available providers are tested.
    """
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    reference = (body.get("reference") or "").strip()
    providers = body.get("providers")  # optional list

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if not reference:
        return jsonify({"error": "Reference answer is required"}), 400

    result = run_benchmark(prompt, reference, providers)
    if "error" in result:
        return jsonify(result), 502
    return jsonify(result)
