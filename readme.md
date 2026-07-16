# Cold Email Generator

Personalized cold emails for founder outreach. Uses Gemma via Gemini API for research + writing, saves drafts directly to your Gmail with your resume attached.

## What it does

For each founder in your CSV, this app:
1. Researches the company using the description + search snippets
2. Writes a personalized email grounded in their actual problems
3. Saves the email as a Gmail draft with your resume attached
4. You review and send from Gmail

## Requirements

- Python 3.10+
- Google account with Gmail
- Gemini API key (free)
- OAuth credentials for Gmail draft access

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/cold-outreach.git
cd cold-outreach
pip install -r requirements.txt
```

### 2. Get a Gemini API key

- Go to https://aistudio.google.com/apikey
- Create a key (free, no card needed)
- Copy it — you'll paste it into the app

### 3. Get Gmail OAuth credentials

**If I sent you `google_client_secrets.json` directly**: save it in the project folder. Send me your Gmail address so I can add you as a test user.

**Otherwise, create your own**:

1. Go to https://console.cloud.google.com/ → create a new project
2. Enable Gmail API: APIs & Services → Library → search "Gmail API" → Enable
3. Configure OAuth consent screen:
   - APIs & Services → OAuth consent screen → External
   - Fill in app name + your email
   - Add scope: `https://www.googleapis.com/auth/gmail.compose`
   - Add your Gmail as a test user
4. Create credentials:
   - APIs & Services → Credentials → Create Credentials → OAuth client ID
   - Application type: **Desktop app**
   - Download JSON → save as `google_client_secrets.json` in the project folder

### 4. Prepare your leads CSV

Required columns:
- `Organization Name`, `Full Name`, `First Name`, `Email`
- `Company Description`, `Status`
- `result_1_title`, `result_1_url`, `result_1_snippet` through `result_5_*`

A sample is in `data/sample_leads.csv`.

## Run it

```bash
streamlit run app.py
```

Opens at http://localhost:8501

Then:
1. Paste your Gemini key in the sidebar
2. Click "Sign in with Google" → approve
3. Enter your full name for the signature
4. Upload your resume (PDF/DOCX/MD/TXT)
5. Upload your leads CSV
6. Set "Test on first N leads" to 3 for your first run
7. Click Generate

Drafts appear in your Gmail as each one finishes. Review and send from Gmail.

## Common issues

| Problem | Fix |
|---|---|
| "Access blocked" on Google sign-in | You need to be added as a test user in the OAuth consent screen |
| "Missing google_client_secrets.json" | Save the OAuth JSON in the project folder with that exact filename |
| Drafts don't save | Click "Disconnect Gmail" in sidebar, sign in again — token may have expired |
| "RESOURCE_EXHAUSTED" | Daily Gemma quota hit (14,400 RPD, resets midnight Pacific) — unlikely at normal use |
| Docling install fails | `pip install docling --no-cache-dir` |

## What data goes where

- Gemini API key → sent to Google's Gemini API for LLM calls
- Company descriptions, search snippets, resume → sent to Gemini as prompts
- Drafts → saved to YOUR Gmail via YOUR OAuth token
- No third party in between

## Free tier limits

- Gemma 3 27B via Gemini API: 14,400 requests/day
- At 2 calls per email = ~7,000 emails/day of capacity
- Way more than you should ever send

## Security

`.env`, `google_client_secrets.json`, and `token.json` are gitignored. Never commit them. Never share `token.json` — it's your live Gmail access.

## License

MIT. Don't spam founders.