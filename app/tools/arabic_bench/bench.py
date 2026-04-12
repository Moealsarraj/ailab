"""Arabic Bench — evaluates AI Arabic responses against a reference answer."""
from app.core.ai import call_ai_json

_SYSTEM = """Expert Arabic NLP evaluator. Score AI responses against references. Critique in Arabic (فصحى). Return ONLY valid JSON."""

_PROMPT_TMPL = """Evaluate this AI Arabic response against the reference.

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
  "hallucination_risk": <0-100, 0=none>,
  "dialect_awareness": <0-100>,
  "cultural_appropriateness": <0-100>,
  "terminology_accuracy": <0-100>,
  "detected_dialect": "<MSA|Gulf|Egyptian|Levantine|Maghrebi|Mixed|Unknown>",
  "strengths": ["<Arabic>", "<Arabic>"],
  "weaknesses": ["<Arabic>"],
  "critique": "<2-3 paragraphs in Arabic فصحى: semantic alignment, grammar notes, terminology, dialect, missing/hallucinated content>"
}}

Scoring: correctness×0.30 + grammar×0.20 + fluency×0.15 + terminology×0.15 + (100-hallucination)×0.10 + dialect×0.05 + cultural×0.05
Verdict: ≥85 Highly Accurate, ≥70 Good, ≥50 Fair, <50 Poor"""


def evaluate_arabic(ai_response: str, reference: str) -> dict:
    """Evaluate an AI Arabic response against a reference answer."""
    prompt = _PROMPT_TMPL.format(
        ai_response=ai_response[:2000],
        reference=reference[:2000],
    )
    try:
        result = call_ai_json(
            [{"role": "user", "content": prompt}],
            system=_SYSTEM,
            max_tokens=1500,
            task_hint="arabic",
        )
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}
