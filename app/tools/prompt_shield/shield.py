"""Prompt Shield — audits AI system prompts for security vulnerabilities."""
from app.core.ai import call_ai, call_ai_json

_SYSTEM = """You are an expert AI red-teamer and prompt security auditor.
You analyze AI system prompts for vulnerabilities that allow prompt injection,
jailbreaks, privilege escalation, data leakage, and policy bypasses.
You think like an attacker: what can a user say to make this prompt misbehave?
Your findings are specific, actionable, and supported by the actual prompt text.
Return ONLY valid JSON — no markdown fences, no preamble."""

_AUDIT_TMPL = """Perform a security audit on the following AI system prompt.

--- SYSTEM PROMPT START ---
{system_prompt}
--- SYSTEM PROMPT END ---

Return a JSON object with EXACTLY these keys:
{{
  "score": <integer 0-100 — vulnerability score; 0=no vulnerabilities, 100=critically exploitable>,
  "risk_level": "<exactly one of: Minimal Risk | Low Risk | Medium Risk | High Risk | Critical Risk>",
  "summary": "<1-2 sentence plain-English summary of the overall security posture>",
  "vulnerabilities": [
    {{
      "title": "<short name for the vulnerability, e.g. 'Privilege Escalation Backdoor'>",
      "severity": "<exactly one of: critical | high | medium | low>",
      "description": "<2-3 sentences: what the vulnerability is, where it appears in the prompt, and how an attacker could exploit it>",
      "remediation": "<2-3 sentences: specific fix recommendation with example language to add or remove>"
    }}
  ],
  "key_changes": [
    {{
      "original": "<exact quote from the prompt that is problematic>",
      "recommended": "<the hardened replacement text>",
      "note": "<one sentence explaining why this change improves security>"
    }}
  ]
}}

Scoring guide:
- 90-100: Critical Risk — direct injection vectors, hardcoded backdoors, auth bypasses present
- 70-89: High Risk — missing critical constraints, weak refusal language, implicit trust issues
- 50-69: Medium Risk — incomplete safety rules, ambiguous scope, leakable configuration
- 20-49: Low Risk — minor gaps in constraints, style issues, minor leakage potential
- 0-19: Minimal Risk — well-structured, constrained, and resistant to common attacks

Rules:
- vulnerabilities: find ALL issues, even subtle ones (1-6 items depending on quality of prompt)
- key_changes: pick the 2-3 most important changes (not every minor tweak)
- If the prompt is already well-hardened, say so in summary and return score <= 15"""

_HARDEN_SYSTEM = """You are an expert AI prompt security engineer.
Rewrite the given system prompt to fix all security vulnerabilities.
Preserve the original intent and structure. Output ONLY the rewritten prompt — no preamble, no explanation, no fences."""

_HARDEN_TMPL = """Rewrite this system prompt to fix all security issues: weak refusal language,
missing constraints, prompt injection vectors, data leakage risks, and ambiguous scope.
Preserve the original intent completely.

ORIGINAL PROMPT:
---
{system_prompt}
---

Output ONLY the complete hardened prompt text."""


def audit_prompt(system_prompt: str) -> dict:
    """Audit a system prompt for security vulnerabilities."""
    truncated = system_prompt[:4000]

    # Call 1: structured audit (JSON) — no hardened_prompt to avoid parse failures
    audit_result = call_ai_json(
        [{"role": "user", "content": _AUDIT_TMPL.format(system_prompt=truncated)}],
        system=_SYSTEM,
        max_tokens=2048,
    )
    if not audit_result or not isinstance(audit_result, dict):
        return {}

    # Call 2: hardened prompt as plain text (avoids JSON escaping of multiline prompt text)
    try:
        hardened = call_ai(
            [{"role": "user", "content": _HARDEN_TMPL.format(system_prompt=truncated)}],
            system=_HARDEN_SYSTEM,
            max_tokens=2048,
        )
        audit_result["hardened_prompt"] = (hardened or "").strip()
    except Exception:
        audit_result["hardened_prompt"] = ""

    return audit_result
