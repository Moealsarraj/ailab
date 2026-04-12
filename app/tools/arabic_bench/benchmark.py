"""Arabic Bench — multi-model benchmark runner.

Sends the same Arabic prompt to multiple AI providers simultaneously,
collects their responses, then evaluates each against the reference.
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.ai import get_available_providers, call_ai_single, call_ai_json

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = """Expert Arabic NLP evaluator. Score AI responses against references. Critique in Arabic. Return ONLY valid JSON."""

_EVAL_PROMPT = """Evaluate this AI Arabic response against the reference. Be concise.

AI RESPONSE:
{ai_response}

REFERENCE:
{reference}

Return JSON:
{{
  "total_score": <0-100>,
  "verdict": "<Highly Accurate|Good|Fair|Poor>",
  "correctness": <0-100>,
  "grammar": <0-100>,
  "fluency": <0-100>,
  "hallucination_risk": <0-100>,
  "dialect_awareness": <0-100>,
  "terminology_accuracy": <0-100>,
  "detected_dialect": "<MSA|Gulf|Egyptian|Levantine|Maghrebi|Mixed|Unknown>",
  "strengths": ["<Arabic>"],
  "weaknesses": ["<Arabic>"],
  "critique": "<1 paragraph Arabic critique>"
}}

Scoring: correctness×0.30 + grammar×0.20 + fluency×0.15 + terminology×0.15 + (100-hallucination)×0.10 + dialect×0.05 + cultural×0.05"""


def _generate_response(provider_name: str, prompt: str) -> dict:
    """Generate an Arabic response from a single provider."""
    start = time.time()
    try:
        response = call_ai_single(
            provider_name,
            [{"role": "user", "content": prompt}],
            max_tokens=1000,
            use_premium=True,
        )
        elapsed = round(time.time() - start, 2)
        return {
            "provider": provider_name,
            "response": response,
            "time_seconds": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        logger.warning("Benchmark: %s failed: %s", provider_name, e)
        return {
            "provider": provider_name,
            "response": None,
            "time_seconds": elapsed,
            "error": str(e),
        }


def _evaluate_response(ai_response: str, reference: str) -> dict:
    """Evaluate a single AI response against reference."""
    prompt = _EVAL_PROMPT.format(
        ai_response=ai_response[:1500],
        reference=reference[:1500],
    )
    try:
        result = call_ai_json(
            [{"role": "user", "content": prompt}],
            system=_EVAL_SYSTEM,
            max_tokens=800,
            task_hint="arabic",
        )
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def run_benchmark(prompt: str, reference: str, providers: list[str] | None = None) -> dict:
    """Run a benchmark across multiple providers.

    1. Send the same prompt to all available providers in parallel.
    2. Evaluate each response against the reference.
    3. Return ranked results.
    """
    available = get_available_providers()
    available_names = {p["name"] for p in available}

    if providers:
        target_providers = [p for p in providers if p in available_names]
    else:
        target_providers = list(available_names)

    if not target_providers:
        return {"error": "No AI providers available for benchmarking"}

    # Step 1: Generate responses in parallel
    responses = []
    with ThreadPoolExecutor(max_workers=len(target_providers)) as pool:
        futures = {
            pool.submit(_generate_response, name, prompt): name
            for name in target_providers
        }
        for future in as_completed(futures):
            responses.append(future.result())

    # Step 2: Evaluate each successful response
    results = []
    for resp in responses:
        if resp["error"] or not resp["response"]:
            results.append({
                "provider": resp["provider"],
                "model": _get_model_name(resp["provider"]),
                "response": None,
                "time_seconds": resp["time_seconds"],
                "error": resp["error"],
                "evaluation": None,
            })
            continue

        evaluation = _evaluate_response(resp["response"], reference)
        results.append({
            "provider": resp["provider"],
            "model": _get_model_name(resp["provider"]),
            "response": resp["response"],
            "time_seconds": resp["time_seconds"],
            "error": None,
            "evaluation": evaluation,
        })

    # Step 3: Sort by score (highest first), errors last
    results.sort(
        key=lambda r: (r["evaluation"] or {}).get("total_score", -1),
        reverse=True,
    )

    return {
        "prompt": prompt,
        "reference": reference,
        "providers_tested": len(results),
        "results": results,
    }


def _get_model_name(provider: str) -> str:
    """Get the premium model name for display."""
    from app.core.ai import _PREMIUM_MODELS
    return _PREMIUM_MODELS.get(provider, provider)
