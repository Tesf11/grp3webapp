# app/prodcat_api.py
# Done by Chng Jialong
from flask import Blueprint, request, jsonify, current_app
from app.hf_loader import HFAdapter
import os
from app.db import get_session 
from app.models import Entry
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re
from datetime import datetime
import time
from app.models import GenAILog 
from sqlalchemy import func


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optional switch via env; defaults to ON if you have an API key
USE_GENAI = str(os.getenv("USE_GENAI", "1")).lower() not in ("0", "false", "", "no") and bool(GEMINI_API_KEY)

# Configure model
gemini_model = None
GENAI_MODEL_NAME = "disabled"

# Configure Gemini model
if USE_GENAI:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        def pick_gemini_model():
            preferred = ["gemini-1.5-flash", "gemini-2.5-flash"]
            gen_cfg = {"response_mime_type": "application/json"}
            available = list(genai.list_models())
            can_generate = {
                m.name for m in available
                if "generateContent" in getattr(m, "supported_generation_methods", [])
            }
            for name in preferred:
                if name in can_generate:
                    print(f"[Gemini] Using model: {name}")
                    return genai.GenerativeModel(name, generation_config=gen_cfg), name
            if can_generate:
                name = sorted(can_generate)[0]
                print(f"[Gemini] Using fallback model: {name}")
                return genai.GenerativeModel(name, generation_config=gen_cfg), name
            raise RuntimeError("No Gemini models available for this API key.")

        gemini_model, GENAI_MODEL_NAME = pick_gemini_model()
    except Exception as e:
        print(f"[GenAI disabled] {e}")
        gemini_model = None
        GENAI_MODEL_NAME = "disabled"


prodcat_api = Blueprint("prodcat_api", __name__)

# Load your model once at blueprint registration
hf_model = None

def _serialize(obj):
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

def _safe_trim(s: str | None, n: int = 240) -> str:
    if not s:
        return ""
    s = str(s).strip()
    return (s[:n] + "…") if len(s) > n else s

def _log_genai(session, *, model: str, prompt_type: str, prompt: str, resp_text: str | None,
               usage, latency_ms: float, status: str = "ok",
               error_code: str | None = None, safety_blocked: bool = False):
    session.add(GenAILog(
        model=model,
        prompt_type=prompt_type,
        prompt_chars=len(prompt or ""),
        response_chars=len(resp_text or "") if resp_text else 0,
        input_tokens=getattr(usage, "input_tokens", None) if usage else None,
        output_tokens=getattr(usage, "output_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_token_count", None) if usage else None,
        latency_ms=float(latency_ms),
        status=status,
        error_code=error_code,
        safety_blocked=bool(safety_blocked),
        prompt_sample=_safe_trim(prompt, 200),
        response_sample=_safe_trim(resp_text, 200) if resp_text else "",
    ))

@prodcat_api.record_once
def load_model(setup_state):
    global hf_model
    model_dir = os.getenv("MODEL_DIR", "app/models/prodcat_model")
    hf_model = HFAdapter(model_dir, max_length=128)

@prodcat_api.post("/api/predict")
def predict():
    data = request.get_json(force=True, silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400
    category = hf_model.predict(title)[0]
    return jsonify({"title": title, "category": category})

@prodcat_api.get("/api/health")
def health():
    return {"ok": True}

@prodcat_api.post("/api/save/entries")
def api_save_entry():
    data = request.get_json(force=True, silent=True) or {}
    required = ["sample_type","company","product_title","status","category"]
    missing = [k for k in required if not str(data.get(k) or "").strip()]
    if missing:
        return jsonify({"error": f"Missing required: {', '.join(missing)}"}), 400

    with get_session() as s:
        obj = Entry(**{k: v for k, v in data.items() if k in Entry.__table__.columns})
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return jsonify({"ok": True, "id": obj.id}), 200

@prodcat_api.get("/api/list/entries")
def api_list_entries():
    limit = int(request.args.get("limit", 50))
    with get_session() as s:
        rows = s.query(Entry).order_by(Entry.id.desc()).limit(limit).all()
        def ser(r): return {c.name: getattr(r, c.name) for c in r.__table__.columns}
        return jsonify([ser(r) for r in rows]), 200
    
# --- UPDATE (partial) ---
@prodcat_api.put("/api/update/<int:row_id>")
def api_update_entry(row_id: int):
    """Partial update. Accepts JSON with any subset of columns.
       UI may send 'type' for the 'category' DB column; we map it here.
    """
    payload = request.get_json(force=True, silent=True) or {}

    # Map UI field 'type' -> DB field 'category'
    if "type" in payload and "category" not in payload:
        payload["category"] = payload.pop("type")

    # Whitelist updatable fields from the model
    # (don’t allow id or server-managed timestamps to be written)
    allowed = {c.name for c in Entry.__table__.columns} - {"id", "created_at"}
    updates = {k: v for k, v in payload.items() if k in allowed}

    if not updates:
        return jsonify({"error": "No valid fields to update."}), 400

    with get_session() as s:
        obj = s.get(Entry, row_id)
        if not obj:
            return jsonify({"error": f"id {row_id} not found"}), 404

        for k, v in updates.items():
            setattr(obj, k, v)
        s.add(obj)          
        s.flush()            # ensure obj has latest values

        return jsonify({"ok": True, "id": obj.id, "row": _serialize(obj)}), 200


# --- DELETE ---
@prodcat_api.delete("/api/delete/<int:row_id>")
def api_delete_entry(row_id: int):
    with get_session() as s:
        obj = s.get(Entry, row_id)
        if not obj:
            return jsonify({"error": f"id {row_id} not found"}), 404
        s.delete(obj)
        # commit happens on context exit
        return jsonify({"ok": True, "id": row_id}), 200
    
def _clip(s, n): 
    s = re.sub(r"\s+"," ", str(s or "").strip())
    return s if len(s) <= n else s[:n-1] + "…"

def _short_code(row):
    title = re.sub(r"[^A-Za-z0-9 ]+","", (row.get("product_title") or "")).strip()
    head = "".join([w[:3] for w in title.split()[:2]]).upper() or "TAG"
    comp = (row.get("company") or "CMP")[:3].upper()
    try:
        dt = datetime.fromisoformat((row.get("eta") or "").strip()).strftime("%m%d")
    except Exception:
        dt = datetime.now().strftime("%m%d")
    track = (row.get("tracking_number") or "NA")[-5:].upper()
    return f"{head}-{comp}-{dt}-{track}"

SYSTEM_SCHEMA = """
Return ONLY strict JSON with keys:

{
  "lang": "en|ms|zh|...",

  // 1-line max 28–34 chars depending on size
  "headline": "VERY short product name, brand + variant if space",

  // <= 2 lines, each <= 30 chars (optional)
  "subhead": ["optional line 1", "optional line 2"],

  // 2–4 short bullets, <= 26 chars each, highest-signal facts:
  // examples: size/weight/pack, color/scent, use/location, quick instruction
  "bullets": ["...", "..."],

  // short handling flags, e.g., "Keep dry", "Fragile", "Keep cool"
  "handling": ["..."],

  // optional hazards like "Irritant", "Flammable" (keep very short)
  "hazards": ["..."],

  // printed footer code on label (client can override)
  // if omitted, we will compute it server-side
  "footer_code": "<optional>",

  // content encoded into QR (compact JSON is fine)
  "qr_payload": {
    "product_title": "...",
    "company": "...",
    "category": "...",
    "tracking_number": "...",
    "eta": "...",
    "contact": "..."
  }
}

Rules:
- Keep all strings as short as possible for a 58–62mm label.
- No markdown, no code fences, ONLY raw JSON.
- Always include 'headline', 'subhead', 'bullets', 'handling', 'qr_payload'.
- If you cannot infer product-specific bullets, use essentials (category, company, status, size/pack if present).
- Never invent pack counts or sizes/volumes if not provided; omit them instead.
- Bullets should not contain guessed quantities like “20 pcs”, “500ml”, “x10”.


"""


def _compose_zpl(plan: dict, code: str, wide: bool = False):
    """
    Improved ZPL layout for ~58–62mm labels (203 dpi).
    Set wide=True if you’re on a 4" printer and want more room.
    """
    # pull fields safely
    headline = _clip(plan.get("headline"), 34 if wide else 30)
    subhead  = [ _clip(x, 30 if wide else 28) for x in (plan.get("subhead") or [])[:2] ]
    bullets  = [ _clip(x, 28 if wide else 26) for x in (plan.get("bullets") or [])[:4] ]
    handling = [ _clip(x, 18) for x in (plan.get("handling") or [])[:3] ]
    hazards  = [ _clip(x, 14) for x in (plan.get("hazards")  or [])[:3] ]
    payload  = plan.get("qr_payload") or {}
    qr_text  = json.dumps(payload, separators=(",",":"))

    # width in dots
    PW = 800 if wide else 600
    LEFT = 30
    RIGHT_QR_X = 430 if not wide else 580

    z = []
    z.append("^XA")
    z.append("^CI28")
    z.append(f"^PW{PW}")
    z.append("^LH0,0")

    # Headline
    z.append("^CF0,46" if not wide else "^CF0,52")
    z.append(f"^FO{LEFT},30^FB{RIGHT_QR_X-LEFT-20},1,0,L,0^FD{headline}^FS")

    # Subhead (smaller)
    y = 90 if not wide else 100
    if subhead:
        z.append("^CF0,30")
        for ln in subhead:
            z.append(f"^FO{LEFT},{y}^FB{RIGHT_QR_X-LEFT-20},1,0,L,0^FD{ln}^FS")
            y += 32

    # Separator line
    z.append(f"^FO{LEFT},{y+6}^GB{RIGHT_QR_X-LEFT-20},1,1^FS")
    y += 20

    # Bullets
    if bullets:
        z.append("^CF0,28")
        for ln in bullets:
            z.append(f"^FO{LEFT},{y}^FB{RIGHT_QR_X-LEFT-20},1,0,L,0^FD• {ln}^FS")
            y += 30

    # Handling / Hazards (small, one line)
    badges = []
    if handling: badges.extend(handling[:2])
    if hazards:  badges.extend(hazards[:2])
    if badges:
        z.append("^CF0,24")
        z.append(f"^FO{LEFT},{y+6}^FB{RIGHT_QR_X-LEFT-20},1,0,L,0^FD" + "  ".join(badges) + "^FS")
        y += 34

    # QR on right
    z.append(f"^FO{RIGHT_QR_X},30^BQN,2,6")
    z.append("^FDLA," + qr_text + "^FS")

    # Footer code
    z.append("^CF0,26")
    z.append(f"^FO{LEFT},{y+12}^FD{code}^FS")

    z.append("^XZ")
    return "\n".join(z)


@prodcat_api.post("/api/generate_smart_tag")
def api_generate_smart_tag():
    if gemini_model is None:
        return jsonify({"error": "GenAI not configured"}), 503


    data = request.get_json(force=True, silent=True) or {}
    if "type" in data and "category" not in data:
        data["category"] = data["type"]

    title = (data.get("product_title") or "").strip()
    if not title:
        return jsonify({"error": "product_title is required"}), 400

    lang = (data.get("lang") or "en").strip()

    prompt = f"""
You are designing a small warehouse sample label.
Goal: make it scannable and legible on a 58–62mm thermal label.
Prioritize: short headline, 0–2 subhead lines, 2–4 concise bullets (size/pack/scent or most useful facts), handling and hazards when relevant.

Language: {lang}
{SYSTEM_SCHEMA}

INPUT:
product_title: {data.get('product_title','')}
category: {data.get('category','')}
company: {data.get('company','')}
sample_type: {data.get('sample_type','')}
status: {data.get('status','')}
tracking_number: {data.get('tracking_number','')}
eta: {data.get('eta','')}
contact_person: {data.get('contact_person','')}
notes: {data.get('notes','')}
""".strip()

    t0 = time.perf_counter()
    raw = ""
    try:
        r = gemini_model.generate_content(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Safety block detection (SDK-specific fields)
        blocked = False
        if getattr(r, "prompt_feedback", None):
            pf = r.prompt_feedback
            blocked = (getattr(pf, "block_reason", None) is not None)

        raw = (r.text or "").strip()
        def extract_json(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
                s = re.sub(r"\s*```$", "", s, flags=re.DOTALL)
                s = s.strip()
            if s.lower().startswith("json\n"):
                s = s.split("\n", 1)[1].strip()
            return s

        cleaned = extract_json(raw)
        try:
            plan = json.loads(cleaned)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise
            plan = json.loads(m.group(0))

        # --- enrich/fallbacks (unchanged from your version) ---
        plan = plan or {}
        headline = _clip(plan.get("headline") or title, 30)
        plan["headline"] = headline

        subhead = plan.get("subhead") or []
        if not isinstance(subhead, list):
            subhead = [str(subhead)]
        if not subhead:
            comp = (data.get("company") or "").strip()
            s1 = (data.get("sample_type") or "").strip()
            stt = (data.get("status") or "").strip()
            joiner = " · ".join([x for x in [comp, s1 or stt] if x])
            if joiner:
                subhead = [_clip(joiner, 28)]
        plan["subhead"] = subhead[:2]

        bullets = plan.get("bullets") or []
        if not isinstance(bullets, list):
            bullets = [str(bullets)]

        def _maybe(s):
            s = (s or "").strip()
            return s if s else None

        canon = []
        if _maybe(data.get("sample_type")):
            canon.append(_clip(f"{data['sample_type']} Sample", 26))
        if _maybe(data.get("status")):
            canon.append(_clip(data["status"], 26))
        if _maybe(data.get("category")):
            canon.append(_clip(data["category"], 26))
        if _maybe(data.get("tracking_number")):
            canon.append(_clip(f"Track: {data['tracking_number'][-6:]}", 26))
        plan["bullets"] = canon[:4]

        handling = plan.get("handling") or []
        if not isinstance(handling, list):
            handling = [str(handling)]
        if not handling:
            handling = ["Keep dry"]
        plan["handling"] = [_clip(h, 18) for h in handling[:3]]

        hazards = plan.get("hazards") or []
        if not isinstance(hazards, list):
            hazards = [str(hazards)]
        plan["hazards"] = [_clip(h, 14) for h in hazards[:3]]

        qp = plan.get("qr_payload", {}) or {}
        qp.setdefault("product_title", data.get("product_title"))
        qp.setdefault("company",        data.get("company"))
        qp.setdefault("category",       data.get("category"))
        qp.setdefault("tracking_number",data.get("tracking_number"))
        qp.setdefault("eta",            data.get("eta"))
        qp.setdefault("contact",        data.get("contact_person"))
        plan["qr_payload"] = qp

        code = _short_code(data)
        if not plan.get("footer_code"):
            plan["footer_code"] = code

        zpl = _compose_zpl(plan, code)

        # --- LOG SUCCESS/blocked ---
        usage = getattr(r, "usage_metadata", None)
        with get_session() as s:
            _log_genai(
                s,
                model=GENAI_MODEL_NAME,
                prompt_type="smart_tag",
                prompt=prompt,
                resp_text=raw,
                usage=usage,
                latency_ms=latency_ms,
                status=("blocked" if blocked else "ok"),
                safety_blocked=blocked,
            )

        if blocked:
            return jsonify({"error": "Request blocked by safety filters"}), 403

        # Markdown preview (unchanged)
        headline = plan.get("headline") or title
        subhead  = plan.get("subhead") or []
        bullets  = plan.get("bullets") or []
        badges   = (plan.get("handling") or [])[:2] + (plan.get("hazards") or [])[:2]
        code     = plan.get("footer_code") or _short_code(data)

        br = "  \n"
        lines = [f"**{headline}**"]
        for sh in subhead: lines.append(sh)
        for b in bullets:  lines.append(f"• {b}")
        if badges:         lines.append("_" + " · ".join(badges) + "_")
        lines.append(f"`{code}`")
        md_preview = br.join(lines)

        return jsonify({
            "ok": True,
            "plan": plan,
            "short_code": code,
            "zpl": zpl,
            "markdown_preview": md_preview
        }), 200

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        with get_session() as s:
            _log_genai(
                s,
                model=GENAI_MODEL_NAME,
                prompt_type="smart_tag",
                prompt=prompt,
                resp_text=None,
                usage=None,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
        return jsonify({"error": f"genai_parse_failed: {e}", "raw": (raw if 'raw' in locals() else "")}), 500


@prodcat_api.get("/api/metrics/genai")
def api_metrics_genai():
    """Lightweight aggregates for last N days (default 7)."""
    days = int(request.args.get("days", 7))
    with get_session() as s:
        base = s.query(GenAILog).filter(GenAILog.ts >= func.datetime("now", f"-{days} days"))

        total   = base.count()
        ok      = base.filter(GenAILog.status == "ok").count()
        blocked = base.filter(GenAILog.status == "blocked").count()
        errors  = base.filter(GenAILog.status == "error").count()

        avg_lat = s.query(func.avg(GenAILog.latency_ms)).filter(GenAILog.ts >= func.datetime("now", f"-{days} days")).scalar() or 0.0

        # SQLite has no percentile_cont; compute p95 in Python if you want (front-end).
        sum_in  = s.query(func.sum(GenAILog.input_tokens)).filter(GenAILog.ts >= func.datetime("now", f"-{days} days")).scalar() or 0
        sum_out = s.query(func.sum(GenAILog.output_tokens)).filter(GenAILog.ts >= func.datetime("now", f"-{days} days")).scalar() or 0

        by_model = (
            s.query(GenAILog.model, func.count(GenAILog.id))
             .filter(GenAILog.ts >= func.datetime("now", f"-{days} days"))
             .group_by(GenAILog.model)
             .all()
        )

        return jsonify({
            "window_days": days,
            "counts": {"total": total, "ok": ok, "blocked": blocked, "errors": errors},
            "latency_ms": {"avg": float(avg_lat)},
            "tokens": {"input": int(sum_in), "output": int(sum_out), "total": int(sum_in + sum_out)},
            "by_model": [{"model": m, "count": c} for (m, c) in by_model],
        }), 200
