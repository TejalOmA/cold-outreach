"""app.py — unified Streamlit UI: resume parse → email gen → Gmail drafts (OAuth).

Uses Google OAuth for Gmail auth (no app passwords).
Uses Gemma 3 27B via Gemini API (14,400 RPD on free tier).

Usage:
    pip install streamlit google-genai pydantic python-dotenv pandas docling google-auth google-auth-oauthlib google-api-python-client
    streamlit run app.py

First-time setup:
    1. Google Cloud Console → OAuth consent screen configured with gmail.compose scope
    2. Add your Gmail as a test user
    3. Create OAuth client (Desktop app type)
    4. Download JSON → save as `google_client_secrets.json` next to this file
    5. Add `google_client_secrets.json` and `token.json` to .gitignore
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from collections import deque
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import pandas as pd
import streamlit as st
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from pydantic import BaseModel, Field

load_dotenv()

# ==============================================================
# CONFIG
# ==============================================================

MODEL_CANDIDATES = [
#    "gemma-3-27b-it",           # 14,400 RPD — massive headroom
    "gemini-3.1-flash-lite",    # 500 RPD backup
    "gemma-3-12b-it",           # 14,400 RPD if 27B has issues
    "gemini-2.5-flash",         # 20 RPD last resort
]

RPM_LIMIT = 25   # stay under 30 RPM (Gemma limit)

BANNED_WORDS = [
    "passionate", "excited", "world-class", "cutting-edge", "innovative",
    "rockstar", "fast-paced", "synergy", "leverage", "space", "vertical",
    "solution", "ai-powered", "revolutionize", "game-changing", "seamless",
    "aspirant", "enthusiast", "driven", "motivated",
]

COL_COMPANY = "Organization Name"
COL_FIRST = "First Name"
COL_FOUNDER = "Full Name"
COL_DESC = "Company Description"
COL_STATUS = "Status"
COL_EMAIL = "Email"

DEFAULT_SUBJECT = "Exploring opportunities at {company}"

# OAuth config
SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
CREDS_FILE = Path("google_client_secrets.json")
TOKEN_FILE = Path("token.json")

# ==============================================================
# GEMINI CLIENT (lazy init + model auto-detection)
# ==============================================================

_client: Optional[genai.Client] = None
_working_model: Optional[str] = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in sidebar")
        _client = genai.Client(api_key=api_key)
    return _client

def get_working_model() -> str:
    global _working_model
    if _working_model:
        return _working_model
    client = get_client()
    errors = []
    for m in MODEL_CANDIDATES:
        try:
            client.models.generate_content(model=m, contents="ok")
            _working_model = m
            print(f"[model] using {m}", flush=True)
            return m
        except Exception as e:
            errors.append(f"{m}: {str(e)[:120]}")
            print(f"[model] {m} unavailable: {str(e)[:120]}", flush=True)
    raise RuntimeError(
        "No working Gemini model found on this API key. Tried:\n" + "\n".join(errors)
    )

def reset_client():
    global _client, _working_model
    _client = None
    _working_model = None

# ==============================================================
# RATE LIMITER
# ==============================================================

_call_times: deque = deque()

def _wait_for_rate_limit():
    now = time.time()
    while _call_times and _call_times[0] < now - 60:
        _call_times.popleft()
    if len(_call_times) >= RPM_LIMIT:
        wait = 60 - (now - _call_times[0]) + 0.5
        time.sleep(wait)
        now = time.time()
        while _call_times and _call_times[0] < now - 60:
            _call_times.popleft()
    _call_times.append(time.time())

# ==============================================================
# LLM CALLS
# ==============================================================

def llm_structured(prompt: str, schema, system: str | None = None, max_retries: int = 3):
    """Works for both Gemini and Gemma models.
    Falls back to prompt-based JSON for Gemma since response_schema may not work there."""
    client = get_client()
    model = get_working_model()
    schema_json = schema.model_json_schema()
    last_error = None
    is_gemma = "gemma" in model.lower()

    for attempt in range(max_retries):
        _wait_for_rate_limit()
        try:
            if is_gemma:
                # Prompt-based JSON extraction
                json_system = (system or "") + (
                    f"\n\nReturn ONLY a valid JSON object matching this schema:\n"
                    f"{json.dumps(schema_json, indent=2)}\n\n"
                    f"Do not include any text before or after the JSON. "
                    f"Do not wrap in markdown code fences."
                )
                config = types.GenerateContentConfig(
                    system_instruction=json_system,
                    temperature=0.4,
                )
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                text = resp.text.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                    if text.endswith("```"):
                        text = text[:-3].strip()
                return schema.model_validate_json(text)
            else:
                # Gemini native structured output
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    system_instruction=system,
                    temperature=0.4,
                )
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                return schema.model_validate_json(resp.text)

        except Exception as e:
            last_error = e
            msg = str(e)
            print(f"[llm_structured attempt {attempt+1}] {type(e).__name__}: {msg[:250]}", flush=True)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(15 * (2 ** attempt))
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"max retries. Last: {type(last_error).__name__}: {last_error}")

def llm_text(prompt: str, system: str | None = None, max_retries: int = 3) -> str:
    client = get_client()
    model = get_working_model()
    last_error = None
    for attempt in range(max_retries):
        _wait_for_rate_limit()
        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.5,
            )
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            return resp.text.strip()
        except Exception as e:
            last_error = e
            msg = str(e)
            print(f"[llm_text attempt {attempt+1}] {type(e).__name__}: {msg[:250]}", flush=True)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(15 * (2 ** attempt))
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"max retries. Last: {type(last_error).__name__}: {last_error}")

# ==============================================================
# SCHEMAS
# ==============================================================

class CompanyProfile(BaseModel):
    one_line_pitch: str
    core_technical_problems: list[str] = Field(default_factory=list, max_length=5)
    recent_specific_things: list[str] = Field(default_factory=list)
    founder_background: str = ""
    hook_material: str

# ==============================================================
# PROMPTS
# ==============================================================

RESEARCH_SYSTEM = """You extract structured facts about a company for a candidate writing a cold email.

CRITICAL RULES:
- Every field must be grounded in the provided sources (company description + search snippets). Do not invent facts.
- 'core_technical_problems' must be ACTUAL engineering problems, not marketing categories.
  BAD: "AI-powered logistics platform"
  GOOD: "extracting structured shipment data from photos of paper delivery orders sent via WhatsApp"
- 'recent_specific_things' should only include things clearly indicated by dates or contextual clues in the sources. If nothing genuinely recent is mentioned, leave empty.
- 'hook_material' is the single strongest thing to reference in a cold email's first line. It must be SPECIFIC — a recent launch, a distinctive technical problem, a distinctive founder detail. Never a category ("AI startup", "email builder", etc).
- 'founder_background' should only be filled if the search results contain concrete details about the founder. Empty otherwise.
- Prefer language from the sources over paraphrase. Do not add adjectives the sources didn't use."""

EMAIL_SYSTEM = """You write cold email bodies from a technical candidate to a startup founder.

VOICE: engineer to engineer. First person. No recruiter language, no hype, no credential-listing.
FOUNDERS ARE NOT PROFESSORS. Do not write formally. Do not open with "I am [name], a [degree from school]." Do not close with "Yours sincerely" or "I assure you." Do not offer references.

STRUCTURE (mandatory paragraph breaks):

Paragraph 1 — Hook (1-2 sentences)
Open with "I came across {company} —" OR "Saw your [specific thing] —" and immediately reference a SPECIFIC named product, feature, recent launch, blog post, or technical problem from hook_material. Follow with one clause about why this candidate finds it interesting, grounded in something they've actually built.

Paragraph 2 — Achievement 1 + Bridge (2-3 sentences)
State one achievement from the resume with metrics preserved EXACTLY. Immediately follow with a bridge clause using this pattern: "[what I did] → maps to / could be extended to / directly applies to → [their specific technical problem]". Name a specific problem from core_technical_problems. Do NOT use vague bridges like "which aligns with your goals" or "which is relevant to your space."

Paragraph 3 — Achievement 2 + Bridge (2-3 sentences)
Same as paragraph 2 with a different achievement and a different specific problem.

Paragraph 4 — Close (1 sentence)
Offer a call or take-home. Be concrete: "Would you be open to a quick call this week?" not "I'd love to chat sometime."

HARD RULES:
- Address by first name only: "Hi Loki,"
- Preserve every number from the resume EXACTLY as written. No rounding, no rephrasing.
- The email must fail the swap test: replacing the company name should make the body no longer make sense.
- BANNED WORDS: passionate, excited, world-class, cutting-edge, innovative, rockstar, fast-paced, synergy, leverage, space, vertical, solution, AI-powered, revolutionize, game-changing, seamless, aspirant, enthusiast, driven, motivated.
- 140-220 words total.
- Return ONLY the email body starting with "Hi {founder_first_name},"
- Do NOT include subject line or signature — those are added programmatically.
- Do not use em dashes.
"""

# ==============================================================
# PIPELINE
# ==============================================================

def extract_snippets_from_row(row: dict) -> list[dict]:
    snippets = []
    for i in range(1, 6):
        title = str(row.get(f"result_{i}_title", "")).strip()
        url = str(row.get(f"result_{i}_url", "")).strip()
        snippet = str(row.get(f"result_{i}_snippet", "")).strip()
        if title or snippet:
            snippets.append({"title": title, "url": url, "snippet": snippet})
    return snippets

def build_research_prompt(company_name: str, description: str, snippets: list[dict]) -> str:
    snip_text = ""
    for i, s in enumerate(snippets, 1):
        if s.get("title") or s.get("snippet"):
            snip_text += f"\n[Search result {i}]\nTitle: {s.get('title', '')}\nURL: {s.get('url', '')}\nSnippet: {s.get('snippet', '')}\n"
    return f"""Company: {company_name}

YC Description (founder-written):
{description}

Search results about the founder and company:
{snip_text or '(no search results)'}

Extract the structured profile."""

def build_email_prompt(founder_first: str, company_name: str, profile: CompanyProfile, resume: str) -> str:
    return f"""Candidate's resume:
---
{resume}
---

Target company: {company_name}
Founder first name: {founder_first}

Company profile:
- One-line pitch: {profile.one_line_pitch}
- Core technical problems:
{chr(10).join(f'  - {p}' for p in profile.core_technical_problems)}
- Recent specific things: {profile.recent_specific_things or '(none)'}
- Founder background: {profile.founder_background or '(unknown)'}
- Hook material (use in opener): {profile.hook_material}

Write the email body."""

def validate_email(body: str, resume_text: str) -> tuple[bool, list[str]]:
    issues = []
    low = body.lower()
    hits = [w for w in BANNED_WORDS if w in low]
    if hits:
        issues.append(f"banned words: {hits}")
    n = len(body.split())
    if n < 130 or n > 230:
        issues.append(f"word count out of range: {n}")
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?(?:/\w+)?\b", body)
    rl = resume_text.lower()
    missing = [x for x in numbers if x.lower() not in rl and x not in ("2","3","4","5")]
    if missing:
        issues.append(f"metrics not in resume: {missing}")
    return len(issues) == 0, issues

def process_lead(row: dict, resume: str) -> tuple[Optional[str], Optional[CompanyProfile], dict]:
    company = str(row.get(COL_COMPANY, "")).strip()
    founder = str(row.get(COL_FOUNDER, "")).strip()
    founder_first = str(row.get(COL_FIRST, "")).strip() or (founder.split()[0] if founder else "")
    description = str(row.get(COL_DESC, "")).strip()

    if not company or not founder or not description:
        return None, None, {"error": "missing required fields"}

    snippets = extract_snippets_from_row(row)
    research_prompt = build_research_prompt(company, description, snippets)
    profile = llm_structured(research_prompt, CompanyProfile, system=RESEARCH_SYSTEM)

    email_prompt = build_email_prompt(founder_first, company, profile, resume)
    body = llm_text(email_prompt, system=EMAIL_SYSTEM)

    ok, issues = validate_email(body, resume)
    if not ok:
        retry_prompt = email_prompt + f"\n\nPrevious draft had issues: {issues}\nFix them."
        body = llm_text(retry_prompt, system=EMAIL_SYSTEM)
        ok, issues = validate_email(body, resume)

    return body, profile, {"ok": ok, "issues": issues}

# ==============================================================
# GOOGLE OAUTH + GMAIL DRAFTS
# ==============================================================

def get_gmail_service():
    """Get authenticated Gmail service. Handles first-run OAuth + token refresh."""
    creds = None

    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            if not CREDS_FILE.exists():
                raise RuntimeError(
                    f"Missing {CREDS_FILE}. Download OAuth credentials from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            # Opens a browser window for the user to consent
            creds = flow.run_local_server(port=0)

        # Save for next time
        TOKEN_FILE.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)

def get_authenticated_email(service) -> str:
    """Return the email address of the authenticated Gmail account."""
    profile = service.users().getProfile(userId="me").execute()
    return profile["emailAddress"]

def build_gmail_message(from_email: str, to_email: str, subject: str, body: str,
                        attachment_bytes: bytes, attachment_filename: str) -> dict:
    """Build a MIME message and encode it for the Gmail API."""
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    name_lower = attachment_filename.lower()
    if name_lower.endswith(".pdf"):
        maintype, subtype = "application", "pdf"
    elif name_lower.endswith(".docx"):
        maintype, subtype = "application", "vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        maintype, subtype = "application", "octet-stream"

    part = MIMEBase(maintype, subtype)
    part.set_payload(attachment_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{attachment_filename}"')
    msg.attach(part)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return {"message": {"raw": raw}}

def save_draft_to_gmail(service, from_email: str, to_email: str, subject: str, body: str,
                        attachment_bytes: bytes, attachment_filename: str) -> tuple[bool, str]:
    """Create a draft via the Gmail API."""
    try:
        msg = build_gmail_message(from_email, to_email, subject, body,
                                   attachment_bytes, attachment_filename)
        result = service.users().drafts().create(userId="me", body=msg).execute()
        return True, f"draft id: {result.get('id', 'unknown')}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"

# ==============================================================
# DOCLING PARSER
# ==============================================================

@st.cache_resource
def get_converter():
    return DocumentConverter()

def parse_resume_to_markdown(uploaded_file) -> str:
    Path("temp").mkdir(exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    temp_path = Path(f"temp/resume{suffix}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    converter = get_converter()
    result = converter.convert(temp_path)
    return result.document.export_to_markdown()

# ==============================================================
# STREAMLIT UI
# ==============================================================

st.set_page_config(page_title="Cold Email Generator", page_icon="✉️", layout="wide")

for key, default in [
    ("running", False),
    ("stop_requested", False),
    ("results", []),
    ("resume_markdown", ""),
    ("resume_bytes", None),
    ("resume_filename", ""),
    ("gmail_service", None),
    ("gmail_email", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("✉️ Cold Email Generator")
st.caption("Parse resume → generate emails → save as Gmail drafts (with resume attached)")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔑 Gemini API Key")
    st.caption("Kept only in your browser session.")

    default_gemini = os.environ.get("GEMINI_API_KEY", "")
    gemini_key = st.text_input("Gemini API key", value=default_gemini, type="password")
    st.markdown("[Get free key →](https://aistudio.google.com/apikey)")

    if gemini_key:
        if os.environ.get("GEMINI_API_KEY") != gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            reset_client()

    if st.button("🔄 Reset Gemini connection"):
        reset_client()
        st.success("Gemini connection reset.")

    st.divider()
    st.header("📧 Gmail")

    if st.session_state.gmail_service:
        st.success(f"✅ Connected as {st.session_state.gmail_email}")
        if st.button("🔌 Disconnect Gmail"):
            st.session_state.gmail_service = None
            st.session_state.gmail_email = None
            if TOKEN_FILE.exists():
                TOKEN_FILE.unlink()
            st.rerun()
    else:
        st.caption("Sign in with Google to save drafts.")
        if st.button("🔐 Sign in with Google"):
            try:
                with st.spinner("Opening browser for Google sign-in..."):
                    service = get_gmail_service()
                    st.session_state.gmail_service = service
                    st.session_state.gmail_email = get_authenticated_email(service)
                st.rerun()
            except Exception as e:
                st.error(f"Auth failed: {e}")

    your_name = st.text_input(
        "Your full name (for signature)",
        value=os.environ.get("YOUR_NAME", ""),
        help="Appears in the email signature. Defaults to Gmail prefix if empty.",
    )

    with st.expander("First-time setup"):
        st.markdown("""
        1. Get a Gemini key (link above)
        2. Google Cloud Console: enable Gmail API, configure OAuth consent screen with `gmail.compose` scope, add yourself as a test user
        3. Create OAuth client (Desktop app), download JSON as `google_client_secrets.json`
        4. Click "Sign in with Google" above
        """)

    st.divider()
    st.header("⚙️ Options")
    skip_acquired = st.checkbox("Skip acquired companies", value=True)
    limit = st.number_input("Test on first N leads (0 = all)", min_value=0, value=5)
    subject_template = st.text_input("Subject template", value=DEFAULT_SUBJECT,
                                     help="Use {company} for the company name.")

# ---------- Main area ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    resume_file = st.file_uploader(
        "Resume (PDF, DOCX, MD, TXT)",
        type=["pdf", "docx", "md", "txt"],
        help="Will be parsed with Docling and attached to every draft.",
    )

    if resume_file:
        cur_name = resume_file.name
        if cur_name != st.session_state.resume_filename:
            with st.spinner("Parsing resume with Docling (may take 20-60s)..."):
                try:
                    md = parse_resume_to_markdown(resume_file)
                    st.session_state.resume_markdown = md
                    resume_file.seek(0)
                    st.session_state.resume_bytes = resume_file.getvalue()
                    st.session_state.resume_filename = cur_name
                    st.success(f"Parsed {cur_name}")
                except Exception as e:
                    st.error(f"Parse failed: {e}")

        if st.session_state.resume_markdown:
            with st.expander("Preview parsed markdown"):
                st.code(st.session_state.resume_markdown[:3000], language="markdown")

with col2:
    st.subheader("2. Upload Leads CSV")
    csv_file = st.file_uploader(
        "leads_with_search.csv",
        type=["csv"],
        help="Must have: Organization Name, Full Name, First Name, Email, Company Description, Status, result_1_* through result_5_*",
    )

    if csv_file:
        try:
            preview_df = pd.read_csv(csv_file)
            csv_file.seek(0)
            st.caption(f"{len(preview_df)} rows. Preview:")
            preview_cols = [c for c in [COL_COMPANY, COL_FOUNDER, COL_EMAIL, COL_STATUS]
                            if c in preview_df.columns]
            st.dataframe(preview_df[preview_cols].head(5), use_container_width=True)
        except Exception as e:
            st.error(f"CSV read failed: {e}")

# ---------- Readiness ----------
inputs_ready = all([
    gemini_key.strip(),
    st.session_state.gmail_service is not None,
    st.session_state.resume_bytes is not None,
    csv_file is not None,
])

if not inputs_ready:
    st.info("👈 Fill in the sidebar (Gemini key + Gmail sign-in) and upload both files to enable Generate.")

# ---------- Buttons ----------
btn1, btn2, _ = st.columns([1, 1, 4])

with btn1:
    if st.button("▶ Generate", type="primary",
                 disabled=(not inputs_ready or st.session_state.running)):
        st.session_state.running = True
        st.session_state.stop_requested = False
        st.session_state.results = []
        st.rerun()

with btn2:
    if st.button("⏹ Stop", disabled=not st.session_state.running):
        st.session_state.stop_requested = True

# ---------- Run loop (LIVE) ----------
if st.session_state.running and inputs_ready:
    csv_file.seek(0)
    df = pd.read_csv(csv_file)

    if skip_acquired and COL_STATUS in df.columns:
        df = df[df[COL_STATUS] != "ACQUIRED"].reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(int(limit))

    total = len(df)
    st.write(f"Processing **{total}** leads...")

    progress = st.progress(0.0)
    status_area = st.empty()

    st.divider()
    st.subheader("Live Results")
    lead_placeholders = [st.empty() for _ in range(total)]

    resume_md = st.session_state.resume_markdown
    resume_bytes = st.session_state.resume_bytes
    resume_filename = st.session_state.resume_filename
    gmail_service = st.session_state.gmail_service
    from_email = st.session_state.gmail_email

    for idx, row in df.iterrows():
        if st.session_state.stop_requested:
            status_area.warning(f"🛑 Stopped by user at {idx+1}/{total}")
            break

        company = str(row.get(COL_COMPANY, "")).strip()
        founder = str(row.get(COL_FOUNDER, "")).strip()
        to_email = str(row.get(COL_EMAIL, "")).strip()

        status_area.info(f"[{idx+1}/{total}] {company} — {founder} — researching + writing...")

        with lead_placeholders[idx].container():
            st.markdown(f"### ⏳ #{idx+1} — {company} — {founder}")
            st.caption("Researching + writing...")

        result_row = {
            "#": idx + 1, "company": company, "founder": founder,
            "to": to_email, "subject": "", "status": "processing",
            "email_body": "", "error_detail": "",
        }

        # --- Generate email ---
        try:
            body, profile, validation = process_lead(row.to_dict(), resume_md)
        except Exception as e:
            result_row["status"] = "❌ generation failed"
            result_row["error_detail"] = str(e)
            st.session_state.results.append(result_row)
            with lead_placeholders[idx].container():
                st.markdown(f"### ❌ #{idx+1} — {company} — {founder}")
                st.caption(f"To: {to_email or '(no email)'}")
                st.error(f"Generation failed: {str(e)[:500]}")
            progress.progress((idx + 1) / total)
            continue

        if body is None:
            result_row["status"] = "⏭ skipped (missing data)"
            st.session_state.results.append(result_row)
            with lead_placeholders[idx].container():
                st.markdown(f"### ⏭ #{idx+1} — {company} — {founder}")
                st.caption("Skipped — missing required fields")
            progress.progress((idx + 1) / total)
            continue

        # --- Compose full email ---
        subject = subject_template.format(company=company)
        result_row["subject"] = subject
        signature_name = your_name.strip() if your_name.strip() else from_email.split("@")[0]
        full_body = body + f"\n\nBest regards,\n{signature_name}\n\nAttached: Resume"
        result_row["email_body"] = full_body

        # Show the written email before Gmail save
        with lead_placeholders[idx].container():
            st.markdown(f"### ✍️ #{idx+1} — {company} — {founder}")
            st.caption(f"To: {to_email or '(no email)'} · Subject: {subject}")
            st.text_area(
                "Email body",
                value=full_body,
                height=320,
                key=f"live_body_{idx}",
                label_visibility="collapsed",
            )
            st.caption("Saving to Gmail Drafts...")

        # --- Save to Gmail ---
        if not to_email:
            result_row["status"] = "⚠ generated, no recipient email"
            st.session_state.results.append(result_row)
            with lead_placeholders[idx].container():
                st.markdown(f"### ⚠ #{idx+1} — {company} — {founder}")
                st.caption(f"Subject: {subject}")
                st.warning("Email written but no recipient address in CSV — not saved to Gmail.")
                st.text_area(
                    "Email body",
                    value=full_body,
                    height=320,
                    key=f"live_body_final_{idx}",
                    label_visibility="collapsed",
                )
            progress.progress((idx + 1) / total)
            continue

        saved, err = save_draft_to_gmail(
            service=gmail_service,
            from_email=from_email,
            to_email=to_email,
            subject=subject,
            body=full_body,
            attachment_bytes=resume_bytes,
            attachment_filename=resume_filename,
        )

        if saved:
            result_row["status"] = "✅ draft saved"
            icon = "✅"
            status_text = "Draft saved to Gmail."
        else:
            result_row["status"] = "❌ draft save failed"
            result_row["error_detail"] = err
            icon = "❌"
            status_text = f"Draft save failed: {err[:200]}"

        st.session_state.results.append(result_row)

        # Final render — replace "saving..." with final state
        with lead_placeholders[idx].container():
            st.markdown(f"### {icon} #{idx+1} — {company} — {founder}")
            st.caption(f"To: {to_email} · Subject: {subject}")
            if saved:
                st.success(status_text)
            else:
                st.error(status_text)
            st.text_area(
                "Email body",
                value=full_body,
                height=320,
                key=f"live_body_final_{idx}",
                label_visibility="collapsed",
            )

        progress.progress((idx + 1) / total)

    st.session_state.running = False
    st.session_state.stop_requested = False

    n_saved = sum(1 for r in st.session_state.results if r["status"].startswith("✅"))
    status_area.success(f"Done. Saved {n_saved} drafts to your Gmail.")

# ---------- Compact summary (bottom of page) ----------
if st.session_state.results and not st.session_state.running:
    st.divider()
    n_saved = sum(1 for r in st.session_state.results if r["status"].startswith("✅"))
    n_failed = sum(1 for r in st.session_state.results if r["status"].startswith("❌"))
    n_skipped = sum(1 for r in st.session_state.results
                    if r["status"].startswith("⏭") or r["status"].startswith("⚠"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Saved", n_saved)
    c2.metric("❌ Failed", n_failed)
    c3.metric("⏭ Skipped", n_skipped)
    with c4:
        df_r = pd.DataFrame(st.session_state.results)
        csv_out = df_r.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_out,
                           file_name="results.csv", mime="text/csv")

    st.info("👉 Also check Gmail Drafts folder to review and send.")
