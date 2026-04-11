"""Agent Builder — generates complete AI agent definitions from descriptions."""
import json, re
from app.core.ai import call_ai

_SENTINEL = "---PROMPT---"

_SYSTEM = f"""You are an expert AI agent architect and prompt engineer.
You design precise, production-ready AI agent specifications from plain English descriptions.
Your tool suggestions are practical and named using real API conventions.
You think carefully about edge cases specific to each use case — not generic failures.

Output format — TWO sections separated by exactly this line on its own line:
{_SENTINEL}

SECTION 1 (before the separator): A single JSON object — no markdown fences, no extra text:
{{
  "agent_name": "<2-4 word professional name>",
  "tools": [
    {{
      "name": "<snake_case>",
      "description": "<one sentence: what this tool does and when to call it>",
      "icon": "<single Material Symbols icon name, e.g. search, database, send>",
      "parameters": "<comma-separated key parameter names>"
    }}
  ],
  "examples": [
    {{ "user": "<realistic user message>", "agent": "<realistic agent reply demonstrating correct behavior>" }}
  ],
  "edge_cases": [
    {{ "title": "<short name>", "description": "<how the agent handles this specific case>" }}
  ]
}}
Rules: tools 3-5 (genuine, not placeholders), examples 2-3 pairs, edge_cases 3-4 domain-specific.

SECTION 2 (after the separator): The complete system prompt for the agent.
  - Markdown headers: # ROLE, # MISSION, # BEHAVIOR, # CONSTRAINTS, # OUTPUT FORMAT
  - 200-300 words. Specific to the use case — no generic boilerplate.
  - Raw text only. No JSON. No fences."""


_PROMPT_TMPL = """Generate a complete AI agent definition from this description:

DESCRIPTION: {description}"""


def build_agent(description: str) -> dict:
    """Generate a complete agent definition from a plain English description."""
    prompt = _PROMPT_TMPL.format(description=description[:1000])
    raw = call_ai(
        [{"role": "user", "content": prompt}],
        system=_SYSTEM,
        max_tokens=4096,
    )
    if not raw:
        return {}

    if _SENTINEL in raw:
        json_part, prompt_part = raw.split(_SENTINEL, 1)

        # Strip accidental fences around JSON
        json_part = json_part.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            meta = json.loads(json_part)
        except json.JSONDecodeError:
            # Try to find the outermost JSON object
            m = re.search(r'\{[\s\S]*\}', json_part)
            meta = {}
            if m:
                try:
                    meta = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        meta["system_prompt"] = prompt_part.strip()
        return meta

    # Sentinel missing — treat whole response as system prompt
    return {
        "agent_name": "AI Agent",
        "system_prompt": raw.strip(),
        "tools": [],
        "examples": [],
        "edge_cases": [],
    }
