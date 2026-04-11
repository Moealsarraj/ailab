"""Prompt Bench — runs prompts against test cases and judges outputs."""
from app.core.ai import call_ai, call_ai_json

_SYSTEM_ADVISOR = """You are an expert prompt engineer. Analyse AI test run results and give specific, actionable advice on how to improve the system prompt. Return ONLY valid JSON — no markdown fences."""

_ADVISOR_SCHEMA = """{
  "recommendations": [
    {
      "issue": "<what is going wrong>",
      "fix": "<exact wording or instruction to add/change in the prompt>",
      "severity": "<high|medium|low>"
    }
  ],
  "improved_prompt": "<full rewritten system prompt incorporating all fixes>"
}"""

_ADVISOR_PROMPT = """CURRENT SYSTEM PROMPT:
{system}

TEST RESULTS:
{results_summary}

Based on the failures and warnings above, return JSON matching this schema exactly:
{schema}

Rules:
- Each recommendation must name a concrete change, not vague advice like "be clearer"
- improved_prompt must be a complete, ready-to-use replacement prompt
- Focus on patterns across multiple failures, not one-off edge cases
- severity: high = causes failures, medium = causes warnings, low = minor improvement"""

_SYSTEM_JUDGE = """You are a strict AI evaluation judge.
Assess how well the actual AI output matches the expected output for the given prompt.
Return ONLY valid JSON — no markdown fences."""

_JUDGE_PROMPT = """SYSTEM PROMPT:
{system}

USER INPUT:
{user_input}

EXPECTED OUTPUT:
{expected}

ACTUAL OUTPUT:
{actual}

Return JSON:
{{
  "status": "<pass|fail|warning>",
  "score": <float 0.0-10.0>,
  "notes": "<one sentence explanation>"
}}
Rules:
- pass (score >= 7.0): actual output meets or exceeds expectations
- warning (score 4.0-6.9): partially meets expectations or direction is right but imprecise
- fail (score < 4.0): misses the mark — wrong format, wrong content, or refusal"""


def run_bench(system_prompt: str, test_cases: list) -> dict:
    """Run system_prompt against all test cases, judge each, return aggregate results.

    test_cases: [{ id, label, user_message, expected }]
    """
    results = []

    for i, case in enumerate(test_cases):
        case_id      = case.get("id") or f"T-{i + 1:03d}"
        label        = (case.get("label") or "").strip()
        user_message = (case.get("user_message") or "").strip()
        expected     = (case.get("expected") or "").strip()

        if not user_message:
            continue

        # ── Run the prompt ──────────────────────────────────────────────────────
        try:
            actual = call_ai(
                [{"role": "user", "content": user_message}],
                system=system_prompt,
                max_tokens=1024,
            )
        except Exception as e:
            results.append({
                "id": case_id, "label": label,
                "status": "fail", "score": 0.0,
                "actual_output": f"[Error: {e}]",
                "notes": "AI call failed during execution",
            })
            continue

        # ── Judge the output ────────────────────────────────────────────────────
        try:
            judge_prompt = _JUDGE_PROMPT.format(
                system=system_prompt[:600],
                user_input=user_message[:400],
                expected=expected[:400],
                actual=actual[:600],
            )
            judgment = call_ai_json(
                [{"role": "user", "content": judge_prompt}],
                system=_SYSTEM_JUDGE,
                max_tokens=256,
            )
        except Exception:
            judgment = {"status": "warning", "score": 5.0, "notes": "Judge unavailable — output non-empty"}

        results.append({
            "id":            case_id,
            "label":         label,
            "status":        judgment.get("status", "warning"),
            "score":         round(float(judgment.get("score", 5.0)), 1),
            "actual_output": actual,
            "notes":         judgment.get("notes", ""),
        })

    total    = len(results)
    passed   = sum(1 for r in results if r["status"] == "pass")
    failed   = sum(1 for r in results if r["status"] == "fail")
    warnings = sum(1 for r in results if r["status"] == "warning")
    scores   = [r["score"] for r in results]

    accuracy    = int(passed / total * 100) if total else 0
    consistency = min(int(sum(scores) / len(scores) * 10), 100) if scores else 0

    return {
        "test_results":      results,
        "consistency_score": consistency,
        "accuracy_score":    accuracy,
        "total_cases":       total,
        "passed":            passed,
        "failed":            failed,
        "warnings":          warnings,
        "failed_scenarios":  [r for r in results if r["status"] == "fail"],
    }


def recommend_fixes(system_prompt: str, run_results: list) -> dict:
    """Analyse run results and suggest specific prompt improvements."""
    non_passing = [r for r in run_results if r.get("status") in ("fail", "warning")]
    if not non_passing:
        return {
            "recommendations": [],
            "improved_prompt": system_prompt,
            "all_passed": True,
        }

    lines = []
    for r in run_results:
        lines.append(
            f"[{r.get('id', '?')}] {r.get('label', '')} | "
            f"status={r.get('status')} score={r.get('score', 0)}/10 | "
            f"notes={r.get('notes', '')} | "
            f"actual={str(r.get('actual_output', ''))[:300]}"
        )
    results_summary = "\n".join(lines)

    prompt = _ADVISOR_PROMPT.format(
        system=system_prompt[:800],
        results_summary=results_summary[:2000],
        schema=_ADVISOR_SCHEMA,
    )

    try:
        data = call_ai_json(
            [{"role": "user", "content": prompt}],
            system=_SYSTEM_ADVISOR,
            max_tokens=1500,
        )
        if not isinstance(data, dict):
            raise ValueError("non-dict response")
        return {
            "recommendations": data.get("recommendations") or [],
            "improved_prompt": data.get("improved_prompt") or system_prompt,
            "all_passed": False,
        }
    except Exception as e:
        return {"error": str(e), "recommendations": [], "improved_prompt": system_prompt}
