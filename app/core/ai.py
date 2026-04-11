"""Multi-provider AI engine. Runtime chain: Groq -> Cerebras -> OpenRouter -> Mistral -> Ollama."""
import json, logging, os, re, requests

logger = logging.getLogger(__name__)
_OLLAMA_BASE = "http://localhost:11434"

_PROVIDER_URLS = {
    "groq":       "https://api.groq.com/openai/v1/chat/completions",
    "cerebras":   "https://api.cerebras.ai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "mistral":    "https://api.mistral.ai/v1/chat/completions",
    "openai":     "https://api.openai.com/v1/chat/completions",
}
_FREE_MODELS = {
    "groq":       "llama-3.1-8b-instant",
    "cerebras":   "llama3.1-8b",
    "openrouter": "google/gemma-3-12b-it:free",
    "mistral":    "mistral-small-latest",
}
_PREMIUM_MODELS = {
    "groq":       "llama-3.3-70b-versatile",
    "cerebras":   "qwen-3-235b-a22b-instruct-2507",
    "openrouter": "google/gemma-3-27b-it:free",
    "mistral":    "mistral-medium-latest",
    "openai":     "gpt-4o-mini",
}
_CHAIN_CFG = [
    {"name": "groq",       "key_env": "GROQ_API_KEY",       "timeout": 30, "extra": {}},
    {"name": "cerebras",   "key_env": "CEREBRAS_API_KEY",   "timeout": 30, "extra": {}},
    {"name": "openrouter", "key_env": "OPENROUTER_API_KEY", "timeout": 45,
     "extra": {"HTTP-Referer": "https://github.com/Moealsarraj", "X-Title": "AI Tools"}},
    {"name": "mistral",    "key_env": "MISTRAL_API_KEY",    "timeout": 40, "extra": {}},
]

# Build the runtime provider list — all providers with valid keys
_PROVIDERS = []
for _p in _CHAIN_CFG:
    _k = os.environ.get(_p["key_env"], "")
    if _k:
        _PROVIDERS.append({
            "name":    _p["name"],
            "url":     _PROVIDER_URLS[_p["name"]],
            "model":   _FREE_MODELS[_p["name"]],
            "key":     _k,
            "timeout": _p["timeout"],
            "extra":   _p["extra"],
        })

# Ollama fallback
_OLLAMA_PROVIDER = None
try:
    _r = requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=3)
    if _r.status_code == 200:
        _installed = [m["name"] for m in _r.json().get("models", [])]
        if _installed:
            _OLLAMA_PROVIDER = {"name": "ollama", "model": _installed[0]}
except Exception:
    pass

_AI_AVAILABLE = bool(_PROVIDERS or _OLLAMA_PROVIDER)

_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_RE_OPEN  = re.compile(r"^```[a-z]*\n?", re.MULTILINE)
_RE_CLOSE = re.compile(r"\n?```$", re.MULTILINE)

def _clean(raw: str) -> str:
    raw = _RE_THINK.sub("", raw).strip()
    raw = _RE_OPEN.sub("", raw)
    return _RE_CLOSE.sub("", raw).strip()

def _post_openai(url, key, model, messages, max_tokens, extra_headers, timeout=60):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    headers.update(extra_headers)
    r = requests.post(url, headers=headers,
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
        timeout=timeout)
    r.raise_for_status()
    return _clean(r.json()["choices"][0]["message"]["content"])

def call_ai(messages: list, system: str = "", max_tokens: int = 2048,
            api_key_row: dict | None = None) -> str:
    if system:
        messages = [{"role": "system", "content": system}] + messages
    # Custom API key path (used by e.g. Wasit/Amin integrations)
    if api_key_row:
        provider = api_key_row.get("provider", "openai")
        key      = api_key_row["key"]
        url      = api_key_row.get("url") or _PROVIDER_URLS.get(provider, "")
        model    = api_key_row.get("model") or _PREMIUM_MODELS.get(provider, "gpt-4o-mini")
        if not url:
            raise ValueError(f"No endpoint known for provider {provider!r}")
        if provider == "claude":
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-4-6", "max_tokens": max_tokens, "messages": messages},
                timeout=60)
            r.raise_for_status()
            return _clean(r.json()["content"][0]["text"])
        return _post_openai(url, key, model, messages, max_tokens, {})
    if not _AI_AVAILABLE:
        raise RuntimeError("No AI provider. Set GROQ_API_KEY or similar in .env")
    # Ollama-only path
    if not _PROVIDERS and _OLLAMA_PROVIDER:
        r = requests.post(f"{_OLLAMA_BASE}/api/chat",
            json={"model": _OLLAMA_PROVIDER["model"], "messages": messages, "stream": False},
            timeout=120)
        r.raise_for_status()
        return _clean(r.json()["message"]["content"])
    # Runtime chain: try each provider, fall back on 429 or transient errors
    last_exc = None
    for prov in _PROVIDERS:
        try:
            return _post_openai(
                prov["url"], prov["key"], prov["model"],
                messages, max_tokens, prov["extra"], prov["timeout"]
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 503, 502):
                logger.debug("Provider %s returned %s, trying next", prov["name"], status)
                last_exc = e
                continue
            raise
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_exc = e
            continue
    # Try Ollama as last resort
    if _OLLAMA_PROVIDER:
        r = requests.post(f"{_OLLAMA_BASE}/api/chat",
            json={"model": _OLLAMA_PROVIDER["model"], "messages": messages, "stream": False},
            timeout=120)
        r.raise_for_status()
        return _clean(r.json()["message"]["content"])
    raise last_exc or RuntimeError("All AI providers failed or rate-limited")

def _repair_json(text: str) -> str:
    """Escape literal control characters inside JSON string values."""
    result = []
    in_str = False
    esc = False
    for c in text:
        if esc:
            result.append(c)
            esc = False
            continue
        if c == '\\' and in_str:
            result.append(c)
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            result.append(c)
            continue
        if in_str and c == '\n':
            result.append('\\n')
            continue
        if in_str and c == '\r':
            result.append('\\r')
            continue
        if in_str and c == '\t':
            result.append('\\t')
            continue
        result.append(c)
    return ''.join(result)

def _extract_json(raw: str):
    """Try progressively harder to extract valid JSON from raw text."""
    raw = raw.strip()
    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Repair literal newlines inside strings then retry
    repaired = _repair_json(raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Find first { or [ then walk to find matching closer
    for source in (repaired, raw):
        for start_ch, end_ch in [('{', '}'), ('[', ']')]:
            idx = source.find(start_ch)
            if idx == -1:
                continue
            depth = 0
            in_str = False
            esc = False
            for i in range(idx, len(source)):
                c = source[i]
                if esc:
                    esc = False
                    continue
                if c == '\\' and in_str:
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == start_ch:
                    depth += 1
                elif c == end_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = source[idx:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
    raise ValueError(f"AI returned non-JSON: {raw[:200]}")

def call_ai_json(messages: list, system: str = "", max_tokens: int = 2048,
                 api_key_row: dict | None = None) -> dict | list:
    raw = call_ai(messages, system=system, max_tokens=max_tokens, api_key_row=api_key_row)
    return _extract_json(raw)
