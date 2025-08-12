import os, json, time, hashlib
from typing import List, Optional, Dict, Any
import google.generativeai as genai

MODEL_NAME_DEFAULT = "gemini-2.5-flash"  # or "gemini-2.5-pro" for deeper reasoning

SYSTEM_RULES = (
    "You are a product ideation assistant for ANS Import & Export.\n"
    "- DO NOT predict storage bins or product_type; a separate model handles that.\n"
    "- Focus ONLY on: possible products, industries, and features/benefits.\n"
    "- Be concise, safety-aware, and practical for B2B.\n"
)

SCHEMA_HINT = {"possible_products":["string","..."],"industries":["string","..."],"features":["string","..."]}

def _get_key():
    return os.getenv("GOOGLE_API_KEY")

def _prompt(description: str) -> str:
    return f"""{SYSTEM_RULES}

Return JSON ONLY with this exact shape:
{json.dumps(SCHEMA_HINT, ensure_ascii=False)}

Sample description:
\"\"\"{description.strip()}\"\"\""""

def _try_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        try:
            s = t[t.find("{"): t.rfind("}") + 1]
            return json.loads(s)
        except Exception:
            return {"possible_products": [], "industries": [], "features": []}

def generate_ideas(
    description: str,
    extra_tags: Optional[List[str]] = None,
    model_name: str = MODEL_NAME_DEFAULT,
    temperature: float = 0.7,
    max_retries: int = 2,
    timeout_s: int = 25,
) -> Dict[str, Any]:
    if not description or not description.strip():
        return {"possible_products": [], "industries": [], "features": []}

    api_key = _get_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable")

    genai.configure(api_key=api_key)
    prompt = _prompt(description)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "response_mime_type": "application/json"},
                request_options={"timeout": timeout_s},
            )
            text = (resp.text or "").strip()
            ideas = _try_json(text)

            if extra_tags:
                tags = [t.strip() for t in extra_tags if t and t.strip()]
                # dedupe while preserving order
                ideas["features"] = list(dict.fromkeys((ideas.get("features") or []) + tags))

            return {
                "possible_products": ideas.get("possible_products", []) or [],
                "industries": ideas.get("industries", []) or [],
                "features": ideas.get("features", []) or [],
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(0.7 * (attempt + 1))
            else:
                raise last_err
