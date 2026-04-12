"""Arabic Bench — evaluates AI Arabic responses against a reference answer."""
from app.core.ai import call_ai_json

_SYSTEM = """You are an expert Arabic NLP evaluator and computational linguistics specialist.
You evaluate Arabic AI responses against gold-standard reference answers with rigorous, objective analysis.
You write critique exclusively in Modern Standard Arabic (فصحى) — formal, precise, and analytical.
You score each dimension independently based on linguistic and semantic merit.
Return ONLY valid JSON — no markdown fences, no preamble."""

_PROMPT_TMPL = """Evaluate the following AI-generated Arabic response against a reference (gold standard) answer.

AI RESPONSE:
{ai_response}

REFERENCE ANSWER:
{reference}

Return a JSON object with EXACTLY these keys:
{{
  "total_score": <integer 0-100 — overall weighted evaluation score>,
  "verdict": "<exactly one of: Highly Accurate | Good | Fair | Poor>",
  "correctness": <integer 0-100 — factual accuracy and semantic coverage vs the reference>,
  "grammar": <integer 0-100 — Arabic grammar, morphology, and syntactic correctness>,
  "fluency": <integer 0-100 — naturalness, readability, and stylistic quality of the Arabic>,
  "hallucination_risk": <integer 0-100 — likelihood of fabricated or unsupported content; 0=none, 100=severe>,
  "dialect_awareness": <integer 0-100 — correct handling of MSA vs dialect; 100=perfect MSA or correct dialect recognition>,
  "cultural_appropriateness": <integer 0-100 — cultural sensitivity and contextual appropriateness for Arab audiences>,
  "terminology_accuracy": <integer 0-100 — correct use of domain-specific Arabic terminology and technical terms>,
  "detected_dialect": "<one of: MSA | Gulf | Egyptian | Levantine | Maghrebi | Mixed | Unknown>",
  "strengths": ["<strength 1 in Arabic>", "<strength 2 in Arabic>", "<strength 3 in Arabic>"],
  "weaknesses": ["<weakness 1 in Arabic>", "<weakness 2 in Arabic>"],
  "critique": "<3-5 paragraph analysis written entirely in Arabic (فصحى). Cover: (1) semantic alignment with the reference, (2) grammatical and stylistic observations, (3) terminology comparison, (4) dialect usage assessment, (5) any missing or hallucinated content. Be specific — reference actual phrases from both texts.>"
}}

Scoring rules:
- total_score: weighted average: correctness×0.30 + grammar×0.20 + fluency×0.15 + terminology_accuracy×0.15 + (100 - hallucination_risk)×0.10 + dialect_awareness×0.05 + cultural_appropriateness×0.05
- verdict thresholds: Highly Accurate ≥85, Good ≥70, Fair ≥50, Poor <50
- hallucination_risk: 0 if every claim in the AI response is supported by the reference or is objectively verifiable; higher values indicate invented facts
- dialect_awareness: 100 if output matches the expected register (MSA for formal, correct dialect when asked)
- terminology_accuracy: score how precisely domain terms are translated/used (legal terms, medical terms, tech terms)
- strengths: list 2-3 specific strong points in Arabic
- weaknesses: list 1-2 specific weaknesses in Arabic (empty array if none)
- critique MUST be in Arabic (Modern Standard Arabic / فصحى) — not English"""


def evaluate_arabic(ai_response: str, reference: str) -> dict:
    """Evaluate an AI Arabic response against a reference answer."""
    prompt = _PROMPT_TMPL.format(
        ai_response=ai_response[:4000],
        reference=reference[:4000],
    )
    try:
        result = call_ai_json(
            [{"role": "user", "content": prompt}],
            system=_SYSTEM,
            max_tokens=3000,
        )
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}
