"""MindForge AI Gradio demo — LoRA inference + cyber-health UI aligned with campaign creative."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from gradio.themes.utils import colors as theme_colors
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.training_results_panel import refresh_results_panel

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "outputs/mindforge-qwen-lora")
DEMO_CASES_PATH = os.getenv("DEMO_CASES_PATH", "data/synthetic/mindforge_demo_cases.json")
# Default high enough for full MindForge JSON in one generation (avoid mid-key cutoff).
# Lower via env if you hit VRAM pressure on small GPUs.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "3072"))

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASSETS_DIR = _REPO_ROOT / "assets"
_COVER_PNG = _ASSETS_DIR / "mindforge-cover.png"
_COVER_JPEG = _ASSETS_DIR / "mindforge-cover.jpeg"

# Match evaluate_base_vs_lora.py so demo and offline eval behave similarly.
SYSTEM = """You are MindForge AI, a mental-health risk review assistant for a synthetic hackathon demo.
Return ONLY one JSON object (no markdown fences, no commentary before or after). Do not diagnose, prescribe, or replace a clinician.
Use conservative safety escalation. Synthetic demo data only; no PHI.

Language: **All string values must be English only** (US English). Do not use Chinese, Japanese, Korean, or other scripts — Qwen must not append bilingual crisis disclaimers.

Safety and tone: Use **neutral clinical escalation language only**. Never name terrorist groups, extremist organizations, foreign conflicts, or unrelated third parties. Never threaten the user or claim a specific group **may kill** them. For CRISIS level, say to contact **local emergency services** or a **crisis line** — do not invent sensational scenarios. Avoid stuttering repetition (e.g. do not write "crisis crisis" or "Crisis risk crisis").

Required shape (match keys and types):
- risk_level: one of LOW, MODERATE, HIGH, CRISIS (uppercase)
- risk_score: integer 0-100
- not_medical_advice: true (boolean)
- escalation_recommendation: object with string fields level, timeframe, reason
- primary_concerns, medication_adherence, side_effect_flags, sleep_mood_flags, safety_flags,
  patient_safe_response, clinician_summary, care_loop_targets, missing_info
"""


def _cover_data_uri() -> str | None:
    """Embed poster as data URI so hero background works on HF Spaces without extra static routes."""
    path = _COVER_PNG if _COVER_PNG.exists() else (_COVER_JPEG if _COVER_JPEG.exists() else None)
    if path is None:
        return None
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f'data:{mime};base64,{b64}'


def build_custom_css() -> str:
    """Poster-aligned palette: navy, violet, magenta, coral — glass panels + gradients."""
    cover = _cover_data_uri()
    hero_bg_image = f',\n    url("{cover}")' if cover else ""

    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

:root {{
  /* Hint native controls (scrollbars, inputs) to dark styling — helps shared / remote links. */
  color-scheme: dark;
  --mindforge-bg: #070713;
  --mindforge-panel: rgba(23, 24, 42, 0.78);
  --mindforge-panel-2: rgba(34, 27, 58, 0.72);
  --mindforge-border: rgba(177, 111, 255, 0.32);
  --mindforge-purple: #8b5cf6;
  --mindforge-magenta: #d946ef;
  --mindforge-coral: #ff7a6b;
  --mindforge-orange: #ff9d5c;
  --mindforge-cyan: #7dd3fc;
  --mindforge-text: #f8f7ff;
  /* Brighter than before so intro copy stays readable on glass panels for all viewers. */
  --mindforge-muted: #ddd8f0;
}}

body,
.gradio-container {{
  background:
    radial-gradient(circle at 20% 10%, rgba(255, 122, 107, 0.18), transparent 28%),
    radial-gradient(circle at 80% 5%, rgba(139, 92, 246, 0.24), transparent 32%),
    radial-gradient(circle at 50% 100%, rgba(217, 70, 239, 0.18), transparent 35%),
    linear-gradient(135deg, #05050c 0%, #0b0b1d 48%, #150923 100%) !important;
  color: var(--mindforge-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}}

.mindforge-hero {{
  position: relative;
  min-height: 280px;
  border-radius: 24px;
  overflow: hidden;
  border: 1px solid rgba(177, 111, 255, 0.35);
  margin: 0 0 1rem 0;
  background:
    linear-gradient(90deg, rgba(7, 7, 19, 0.82), rgba(24, 9, 43, 0.68)){hero_bg_image};
  background-size: cover;
  background-position: center;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45), 0 0 60px rgba(139, 92, 246, 0.18);
}}

.hero-overlay {{
  padding: 2rem;
  min-height: 280px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  backdrop-filter: blur(2px);
}}

.hero-kicker {{
  letter-spacing: 0.38em;
  font-size: 0.78rem;
  font-weight: 700;
  color: rgba(248, 247, 255, 0.92);
  text-transform: uppercase;
}}

.hero-subkicker {{
  letter-spacing: 0.32em;
  font-size: 0.82rem;
  font-weight: 800;
  color: var(--mindforge-coral);
  text-transform: uppercase;
  margin-top: 0.35rem;
}}

.hero-main {{
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 2rem;
}}

.hero-brain {{
  width: 74px;
  height: 74px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  font-size: 3rem;
  color: #fff;
  background: radial-gradient(circle, rgba(255, 122, 107, 0.92), rgba(139, 92, 246, 0.84));
  box-shadow: 0 0 36px rgba(217, 70, 239, 0.65);
}}

.mindforge-hero h1 {{
  margin: 0;
  font-size: clamp(2rem, 5vw, 3.2rem);
  line-height: 0.92;
  letter-spacing: 0.12em;
  font-weight: 900;
  color: #ffffff;
  text-shadow: 0 0 28px rgba(139, 92, 246, 0.45);
}}

.mindforge-hero h1 span {{
  background: linear-gradient(90deg, var(--mindforge-magenta), var(--mindforge-coral), var(--mindforge-orange));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}

.hero-tagline {{
  margin: 0.8rem 0 0 0;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: #d8b4fe;
  font-weight: 800;
  font-size: 0.85rem;
}}

.hero-description {{
  max-width: 820px;
  margin-top: 1.2rem;
  color: var(--mindforge-muted);
  font-size: 1rem;
  line-height: 1.55;
}}

.tech-badges {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  margin-top: 1.2rem;
}}

.tech-badges span {{
  border: 1px solid rgba(177, 111, 255, 0.38);
  background: rgba(9, 10, 28, 0.72);
  color: #eee9ff;
  border-radius: 999px;
  padding: 0.45rem 0.8rem;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}

.mindforge-status {{
  border: 1px solid rgba(125, 211, 252, 0.28);
  background: linear-gradient(135deg, rgba(125, 211, 252, 0.12), rgba(139, 92, 246, 0.12));
  border-radius: 16px;
  padding: 0.9rem 1rem;
  color: #f8f7ff;
  font-weight: 700;
  font-size: 0.92rem;
  letter-spacing: 0.02em;
}}

.mindforge-safety {{
  margin-top: 1rem;
  border-left: 4px solid var(--mindforge-coral);
  background: rgba(255, 122, 107, 0.10);
  border-radius: 14px;
  padding: 0.85rem 1rem;
  color: #ffe9e5;
  font-size: 0.92rem;
  line-height: 1.5;
}}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {{
  background: var(--mindforge-panel) !important;
  border: 1px solid var(--mindforge-border) !important;
  border-radius: 18px !important;
  box-shadow: 0 10px 36px rgba(0, 0, 0, 0.26);
  overflow: visible !important;
}}

/* Analyze tab — status column: padding + heading metrics so "### …" is not clipped */
.mindforge-analyze-status-wrap {{
  overflow: visible !important;
  padding: 0.75rem 0.85rem 1rem 0.85rem !important;
}}
.mindforge-analyze-status-wrap .gradio-markdown,
.mindforge-analyze-status-wrap .prose {{
  overflow: visible !important;
  padding: 0.35rem 0.5rem 0.5rem 0.5rem !important;
}}
.mindforge-analyze-status-wrap h1,
.mindforge-analyze-status-wrap h2,
.mindforge-analyze-status-wrap h3 {{
  margin-top: 0.4rem !important;
  margin-bottom: 0.55rem !important;
  padding-top: 0.35rem !important;
  line-height: 1.45 !important;
  color: #ffffff !important;
}}
.mindforge-analyze-status-wrap p,
.mindforge-analyze-status-wrap em,
.mindforge-analyze-status-wrap strong {{
  margin-top: 0.4rem !important;
  line-height: 1.65 !important;
  color: #e8e4f8 !important;
}}
.mindforge-analyze-status-wrap strong {{
  color: #ffffff !important;
}}

/* Tab bar: Soft theme used light `--background-fill-secondary` for hover → white flash for non-dark clients. */
.gradio-container .tab-wrapper button,
.gradio-container .tab-container button {{
  color: #e8e4f8 !important;
  background-color: transparent !important;
}}
.gradio-container .tab-wrapper button:hover:not(:disabled):not(.selected),
.gradio-container .tab-container button:hover:not(:disabled):not(.selected) {{
  background-color: rgba(139, 92, 246, 0.28) !important;
  color: #ffffff !important;
}}

/* Section markdown (e.g. "Risk review output") */
.gradio-container .prose h4 {{
  color: #f4f2ff !important;
  font-weight: 700 !important;
}}

/* Read-only output fields (Care Team Summary, etc.) */
.mindforge-output-field textarea,
.mindforge-output-field input {{
  padding: 0.95rem 1.05rem !important;
  line-height: 1.6 !important;
  min-height: 3.2rem !important;
}}

/* Block labels: theme tokens should match; this backs up contrast if a browser strips variables. */
label,
.label-wrap,
span[data-testid="block-info"] {{
  color: #f8f5ff !important;
  font-weight: 700 !important;
}}
.gradio-container .block-label,
.gradio-container span.block-label-text {{
  background: rgba(34, 28, 61, 0.98) !important;
  color: #f4f0ff !important;
  border: 1px solid rgba(177, 111, 255, 0.35) !important;
}}

.gradio-container button.primary,
.gradio-container button[variant="primary"] {{
  background: linear-gradient(90deg, var(--mindforge-purple), var(--mindforge-magenta), var(--mindforge-coral)) !important;
  border: 0 !important;
  border-radius: 14px !important;
  color: white !important;
  font-weight: 900 !important;
  letter-spacing: 0.05em;
  box-shadow: 0 0 28px rgba(217, 70, 239, 0.32);
}}

textarea,
input,
select {{
  background: rgba(11, 12, 31, 0.82) !important;
  color: #f8f7ff !important;
  border-color: rgba(177, 111, 255, 0.28) !important;
}}
.gradio-container option {{
  background: #14122a !important;
  color: #f8f7ff !important;
}}

.markdown.prose,
.gradio-markdown p {{
  color: var(--mindforge-muted) !important;
}}

.code_wrap,
pre,
code {{
  background: rgba(8, 8, 22, 0.92) !important;
  border-color: rgba(177, 111, 255, 0.28) !important;
}}

footer {{ visibility: hidden; height: 0; }}

/* --- Training results tab: stronger contrast + spacing (mobile / shared links) --- */
#mindforge-tab-training .gradio-markdown.mindforge-results-intro p,
#mindforge-tab-training .mindforge-results-intro p {{
  color: #f0ecff !important;
  font-size: 1.02rem !important;
  line-height: 1.65 !important;
  padding: 0.5rem 0.25rem 1rem 0.25rem !important;
}}

#mindforge-tab-training .mindforge-results-md {{
  padding: 1.25rem 1.5rem !important;
  border-radius: 16px !important;
  background: rgba(12, 14, 36, 0.92) !important;
  border: 1px solid rgba(177, 111, 255, 0.35) !important;
}}

#mindforge-tab-training .mindforge-results-md table {{
  width: 100% !important;
  border-collapse: collapse !important;
  margin: 0.75rem 0 !important;
}}

#mindforge-tab-training .mindforge-results-md th,
#mindforge-tab-training .mindforge-results-md td {{
  padding: 0.65rem 0.85rem !important;
  border-bottom: 1px solid rgba(177, 111, 255, 0.22) !important;
  color: #f5f2ff !important;
  font-size: 0.98rem !important;
}}

#mindforge-tab-training .mindforge-results-md h2 {{
  color: #ffffff !important;
  margin-top: 1.25rem !important;
  font-size: 1.2rem !important;
}}

#mindforge-tab-training .mindforge-results-md p,
#mindforge-tab-training .mindforge-results-md li {{
  color: #e4dff7 !important;
  line-height: 1.7 !important;
}}

#mindforge-tab-training .mindforge-results-plot,
#mindforge-tab-training .mindforge-results-plot > div,
#mindforge-tab-training .mindforge-results-plot .image-container {{
  min-height: 360px !important;
  padding: 1rem 1.25rem 1.5rem 1.25rem !important;
}}

#mindforge-tab-training .block.mindforge-results-plot {{
  padding: 1.2rem 1.4rem 1.6rem 1.4rem !important;
}}

#mindforge-tab-training label,
#mindforge-tab-training .label-wrap {{
  font-size: 1rem !important;
  color: #ffffff !important;
}}
"""


def hero_html() -> str:
    return """
<div class="mindforge-hero">
  <div class="hero-overlay">
    <div class="hero-kicker">BUILDING AI THAT UNDERSTANDS</div>
    <div class="hero-subkicker">EMPOWERING BETTER MENTAL HEALTH</div>
    <div class="hero-main">
      <div class="hero-brain">◐</div>
      <div>
        <h1>MINDFORGE <span>AI</span></h1>
        <p class="hero-tagline">Understand · Insight · Impact</p>
      </div>
    </div>
    <p class="hero-description">
      Mental Health Intelligence.
    </p>
    <div class="tech-badges">
      <span>AMD Instinct™ MI300X</span>
      <span>ROCm</span>
      <span>Qwen Models</span>
      <span>Hugging Face</span>
      <span>Open Source</span>
    </div>
  </div>
</div>
"""


def load_cases():
    path = Path(DEMO_CASES_PATH)
    if not path.exists():
        return {"Custom case": {"patient_note": "Patient missed two doses and slept 3 hours.", "device_or_app_events": {}}}
    obj = json.loads(path.read_text())
    cases = obj.get("cases", obj) if isinstance(obj, dict) else obj
    result = {}
    for c in cases:
        label = c.get("title") or c.get("case_id") or f"Case {len(result)+1}"
        result[label] = c
    return result


CASES = load_cases()

print("Loading model", BASE_MODEL, ADAPTER_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
try:
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
except Exception as e:
    print("Adapter load failed, using base model only:", e)
    model = base
model.eval()


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if "```" not in t:
        return t
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def first_balanced_json_object(text: str) -> str | None:
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        in_string = False
        escape = False
        for j in range(start, len(text)):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]
    return None


_CJK_RE_START = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Hallucinated or harmful model outputs — replace with vetted demo-safe copy (never show to end users).
_UNSAFE_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bisis\b", re.I),
    re.compile(r"\bisil\b", re.I),
    re.compile(r"\bal[- ]?qaeda\b", re.I),
    re.compile(r"\bterrorist(s)?\b", re.I),
    re.compile(r"\bterrorism\b", re.I),
    re.compile(r"may kill you", re.I),
    re.compile(r"will kill you", re.I),
    re.compile(r"\bkill you\b", re.I),
)
_SAFE_PATIENT_REPLACE = (
    "If you might harm yourself or are in immediate danger, contact your local emergency number "
    "or a 24/7 crisis line now. This interface is a synthetic demo and not emergency care."
)
_SAFE_CLINICIAN_REPLACE = (
    "Synthetic demo: elevated risk flags in structured review — route to licensed care per protocol; "
    "use emergency services when indicated. Prior model wording was removed as inappropriate for this demo."
)
_SAFE_GENERIC_REPLACE = "Use neutral escalation language only; no named threats or groups."
# Shown when raw decoder text matches unsafe patterns so the accordion never displays threats or group names.
_RAW_PREVIEW_OMITTED = (
    "(Raw model output omitted: disallowed wording was detected. "
    "Use the structured fields above or retry with a shorter check-in note.)"
)


def _model_output_is_unsafe(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    return any(p.search(s) for p in _UNSAFE_OUTPUT_PATTERNS)


def _compact_crisis_redundancy(s: str) -> str:
    """Remove repetitive 'crisis' stacks and awkward 'crisis: crisis' patterns from model slop."""
    if not isinstance(s, str) or not s.strip():
        return s
    t = s.strip()
    # "Crisis update: Crisis risk crisis:" → one clear crisis lead-in (models love this template).
    t = re.sub(r"(?i)\bcrisis\s+update\s*:\s*", "Crisis: ", t)
    t = re.sub(r"(?i)(\bcrisis\b\s*:\s*)+", "Crisis: ", t)
    t = re.sub(r"(?i)\bcrisis(\s+risk)?\s+crisis\b", "Crisis-level risk", t)
    t = re.sub(r"(?i)(\bcrisis\b)(\s+\1\b)+", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _english_leading_portion(s: str) -> str:
    """
    Keep only the leading English segment before any CJK character.
    Qwen2.5 often appends Chinese crisis / hotline boilerplate after English text.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return s
    m = _CJK_RE_START.search(s)
    if not m:
        return s
    return s[: m.start()].strip().rstrip(" ;:,.—-")


def _sanitize_parsed_for_display(parsed: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy: block unsafe hallucinations, strip CJK tails, reduce crisis word-soup."""
    out = json.loads(json.dumps(parsed))

    def _pipe_patient_clinician(v: str, role: str) -> str:
        if not isinstance(v, str):
            return v
        t = _compact_crisis_redundancy(v)
        if _model_output_is_unsafe(t):
            return _SAFE_PATIENT_REPLACE if role == "patient" else _SAFE_CLINICIAN_REPLACE
        eng = _english_leading_portion(t)
        if eng:
            t = eng
        elif _CJK_RE_START.search(t):
            return (
                "English-only output expected. Retry analysis or shorten the note; "
                "model appended non-English text."
            )
        return _compact_crisis_redundancy(t)

    for key, role in (("patient_safe_response", "patient"), ("clinician_summary", "clinician")):
        v = out.get(key)
        if isinstance(v, str):
            out[key] = _pipe_patient_clinician(v, role)

    er = out.get("escalation_recommendation")
    if isinstance(er, dict):
        for fk in ("level", "timeframe", "reason"):
            vv = er.get(fk)
            if isinstance(vv, str):
                t = _compact_crisis_redundancy(vv)
                if _model_output_is_unsafe(t):
                    er[fk] = _SAFE_GENERIC_REPLACE
                else:
                    c = _english_leading_portion(t)
                    er[fk] = _compact_crisis_redundancy(c if c else t)

    pc = out.get("primary_concerns")
    if isinstance(pc, list):
        cleaned_pc: list[str] = []
        for x in pc:
            sx = _compact_crisis_redundancy(str(x).strip())
            if _model_output_is_unsafe(sx):
                cleaned_pc.append("Safety-related concern (wording redacted for demo)")
            else:
                cleaned_pc.append(_english_leading_portion(sx) or sx)
        out["primary_concerns"] = cleaned_pc

    return out


def _repair_trailing_commas(s: str) -> str:
    """Remove illegal trailing commas before } or ] (common LLM JSON glitch)."""
    return re.sub(r",(\s*[}\]])", r"\1", s)


def extract_json(text: str) -> tuple[str | None, dict[str, Any] | None]:
    """
    Best-effort parse: code fences → first `{` → balanced-brace object → first/last `{`…`}` slice.
    Tries trailing-comma repair. Aligns with offline eval robustness for judge demos.
    """
    cleaned = _strip_code_fences(text.strip())
    if not cleaned:
        return None, None
    # Drop chatter before the opening brace (e.g. "Here is the JSON:").
    brace0 = cleaned.find("{")
    if brace0 > 0:
        cleaned = cleaned[brace0:]

    candidates: list[str] = []
    bal = first_balanced_json_object(cleaned)
    if bal:
        candidates.append(bal)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        bracket_slice = cleaned[start : end + 1]
        if bracket_slice not in candidates:
            candidates.append(bracket_slice)

    last_tried: str | None = None
    for c in candidates:
        for variant in (c, _repair_trailing_commas(c)):
            last_tried = variant
            try:
                return variant, json.loads(variant)
            except json.JSONDecodeError:
                continue
    return last_tried, None


def _risk_score_display(parsed: dict[str, Any]) -> str:
    if "risk_score" in parsed and parsed["risk_score"] is not None:
        return str(parsed["risk_score"])
    band = parsed.get("risk_score_band")
    if isinstance(band, (list, tuple)) and len(band) >= 2:
        return f"{band[0]}–{band[1]} (band)"
    return "—"


def _patient_safe_display(parsed: dict[str, Any]) -> str:
    for key in ("patient_safe_response", "full_summary", "case_summary", "summary"):
        v = parsed.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    ra = parsed.get("recommended_action")
    if isinstance(ra, dict):
        for key in ("patient_safe_response", "medical_advice", "patient_summary"):
            v = ra.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return "—"


def _clinician_display(parsed: dict[str, Any]) -> str:
    for key in ("clinician_summary", "clinician_notes"):
        v = parsed.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    ra = parsed.get("recommended_action")
    if isinstance(ra, dict):
        cs = ra.get("clinician_summary")
        if isinstance(cs, str) and cs.strip():
            return cs.strip()
    return "—"


def _escalation_blob(parsed: dict[str, Any]) -> Any:
    esc = parsed.get("escalation_recommendation")
    if esc is not None:
        return esc
    ra = parsed.get("recommended_action")
    if isinstance(ra, dict):
        inner = ra.get("escalation_recommendation")
        if inner is not None:
            return inner
        return {k: v for k, v in ra.items() if k in ("level", "reason", "timing", "contacts")}
    lv = parsed.get("escalation_level")
    if lv is not None:
        return {"escalation_level": lv}
    return {}


def build_case(case_label: str, free_text: str) -> dict[str, Any]:
    case = CASES.get(case_label, {})
    if free_text and free_text.strip():
        case = {
            "case_id": "CUSTOM-DEMO",
            "context_type": "synthetic_demo",
            "patient_note": free_text.strip(),
            "device_or_app_events": {
                "doses_scheduled": 14,
                "doses_taken": 12,
                "missed_doses_7d": 2,
                "sleep_hours_last_night": 4,
                "mood_score_1_to_10": 3,
            },
            "task": "Return a structured mental-health risk review JSON for the care loop.",
        }
    return case


def run(case_label: str, free_text: str):
    case = build_case(case_label, free_text)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(case)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    tail_preview = raw.strip()[-4000:] if len(raw) > 4000 else raw.strip()
    if _model_output_is_unsafe(raw):
        tail_preview = _RAW_PREVIEW_OMITTED

    candidate, parsed = extract_json(raw)
    if parsed:
        disp = _sanitize_parsed_for_display(parsed)
        risk = str(disp.get("risk_level", "UNKNOWN")).strip() or "UNKNOWN"
        score = _risk_score_display(disp)
        patient = _patient_safe_display(disp)
        clinician = _clinician_display(disp)
        esc = _escalation_blob(disp)
        esc_str = json.dumps(esc, indent=2) if esc else "{}"
        pretty = json.dumps(disp, indent=2)
        status = (
            "### Human Review Ready\n"
            "Structured JSON parsed. Fields below are normalized from model output "
            "(some scenarios use alternate key names)."
        )
        return status, risk, score, esc_str, patient, clinician, pretty, tail_preview

    hint = (
        "The model reply was not valid JSON after extraction. Common causes: **extra prose** around the object, "
        "**markdown fences** only half-closed, **truncation** (raise **`MAX_NEW_TOKENS`** env and restart), or a "
        "**syntax error** inside the object. Raw output is in **Structured JSON** below for debugging."
    )
    status = f"### Needs human review\n{hint}\n\n**Tip:** Retry **Analyze** once; shorter check-in notes reduce load."
    esc_str = "{}"
    fallback_json = candidate if candidate else raw.strip()
    if _model_output_is_unsafe(fallback_json):
        fallback_json = _RAW_PREVIEW_OMITTED
    return (
        status,
        "—",
        "—",
        esc_str,
        "—",
        "—",
        fallback_json,
        tail_preview,
    )


# Soft theme defaults are *light* (white blocks, primary_100 labels). Many remote browsers never
# flip Gradio's `.dark` variables, which produced white label pills + forced white label text →
# invisible copy, and light tab hovers. We set light :root tokens to match the navy UI so every
# client gets consistent contrast without relying on prefers-color-scheme.
_MINDFORGE_THEME = gr.themes.Soft(
    primary_hue=theme_colors.violet,
    secondary_hue=theme_colors.orange,
    neutral_hue=theme_colors.zinc,
).set(
    body_background_fill="#070713",
    background_fill_primary="rgba(14, 14, 28, 0.96)",
    background_fill_secondary="rgba(55, 48, 92, 0.5)",
    body_text_color="#eae6f8",
    body_text_color_subdued="#b8b0d4",
    border_color_primary="rgba(140, 120, 190, 0.45)",
    block_background_fill="rgba(23, 24, 42, 0.78)",
    block_label_background_fill="rgba(34, 28, 61, 0.98)",
    block_label_text_color="#f4f0ff",
    block_title_background_fill="rgba(34, 28, 61, 0.98)",
    block_title_text_color="#f4f0ff",
    input_background_fill="rgba(11, 12, 31, 0.94)",
    input_border_color="rgba(120, 100, 180, 0.45)",
    button_secondary_background_fill="rgba(28, 26, 52, 0.88)",
    button_secondary_text_color="#eae6f8",
    button_secondary_background_fill_hover="rgba(139, 92, 246, 0.35)",
)

with gr.Blocks(title="MindForge AI") as demo:
    gr.HTML(hero_html())
    gr.HTML(
        '<div class="mindforge-status">'
        "Human-in-the-loop review · Privacy-aware workflow · Fine-tuned Qwen on AMD ROCm · "
        "See the <strong>Training results</strong> tab for data mix &amp; offline eval charts"
        "</div>"
    )
    with gr.Tabs():
        with gr.Tab("Analyze"):
            with gr.Column(elem_classes=["mindforge-analyze-status-wrap"]):
                status_md = gr.Markdown(
                    "*Select a **Risk Scenario** or add a **Check-In Note**, then run analysis.*"
                )

            with gr.Row(equal_height=True):
                case_label = gr.Dropdown(
                    list(CASES.keys()),
                    value=list(CASES.keys())[0],
                    label="Risk Scenario",
                    scale=1,
                )
                free_text = gr.Textbox(
                    label="Check-In Note",
                    placeholder="Optional custom vignette — or leave empty to use the scenario above.",
                    lines=6,
                    scale=2,
                )

            run_btn = gr.Button("Analyze Distress Signals", variant="primary", size="lg")

            gr.Markdown("#### Risk review output")

            with gr.Row():
                risk = gr.Textbox(
                    label="Risk Review Level",
                    interactive=False,
                    elem_classes=["mindforge-output-field"],
                )
                score = gr.Textbox(
                    label="Risk Score",
                    interactive=False,
                    elem_classes=["mindforge-output-field"],
                )

            escalation = gr.Code(label="Escalation Pathway", language="json", lines=8)
            patient = gr.Textbox(
                label="Patient-Safe Message",
                lines=5,
                interactive=False,
                elem_classes=["mindforge-output-field"],
            )
            clinician = gr.Textbox(
                label="Care Team Summary",
                lines=5,
                interactive=False,
                elem_classes=["mindforge-output-field"],
            )

            with gr.Accordion("Audit Trail: Structured JSON + Raw Output", open=False):
                raw_json = gr.Code(label="Structured JSON (or unparsed fragment)", language="json", lines=14)
                raw_tail = gr.Textbox(label="Raw generation (truncated for debugging)", lines=6)

            gr.HTML(
                '<div class="mindforge-safety">'
                "<strong>Human review:</strong> MindForge AI does not diagnose, treat, or replace licensed care. "
                "It organizes distress signals and suggests escalation pathways for human review."
                "</div>"
            )

            outputs_list = [status_md, risk, score, escalation, patient, clinician, raw_json, raw_tail]
            run_btn.click(
                fn=run,
                inputs=[case_label, free_text],
                outputs=outputs_list,
                show_progress="full",
            )

        with gr.Tab("Training results"):
            with gr.Column(elem_id="mindforge-tab-training"):
                gr.Markdown(
                    elem_classes=["mindforge-results-intro"],
                    value=(
                        "**Training results** pull from `outputs/training_results_summary.json` "
                        "(set **`TRAINING_RESULTS_JSON`** to override). Rebuild after eval with "
                        "`python scripts/build_training_results_summary.py`, then **Refresh results**."
                    ),
                )
                results_md = gr.Markdown(elem_classes=["mindforge-results-md"])
                dist_plot = gr.Plot(
                    label="Scale training data — risk mix",
                    elem_classes=["mindforge-results-plot"],
                )
                cmp_plot = gr.Plot(
                    label="Structured task adherence — Core schema & escalation (Base vs +LoRA)",
                    elem_classes=["mindforge-results-plot"],
                )
                refresh_btn = gr.Button("Refresh results", variant="secondary")
                refresh_outputs = [results_md, dist_plot, cmp_plot]
                refresh_btn.click(fn=refresh_results_panel, outputs=refresh_outputs)

    demo.load(fn=refresh_results_panel, outputs=refresh_outputs)


def launch() -> None:
    # GRADIO_SHARE=1 creates a temporary https://*.gradio.live URL (no inbound firewall rule needed).
    share_public = os.getenv("GRADIO_SHARE", "").strip().lower() in ("1", "true", "yes")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=_MINDFORGE_THEME,
        css=build_custom_css(),
        # Nudges the browser toward dark form controls when users open the public / proxied URL.
        head='<meta name="color-scheme" content="dark" />',
        allowed_paths=[str(_ASSETS_DIR), str(_REPO_ROOT)],
        share=share_public,
    )


if __name__ == "__main__":
    launch()
