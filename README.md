# CareFlow AI — Web Workstation

CareFlow AI is a human-in-the-loop clinical information coordination and triage-support web application. It accepts multilingual audio, clinical documents, and typed messages; extracts provenance-linked facts; applies a deterministic demonstration triage rule set; and gives clinicians and patients separate, role-appropriate views.

The application is a FastAPI backend with a responsive static web client. All clinical AI services are provided by **Sarvam AI**:

- **GLM-5.3** for structured clinical fact extraction and advisory scoring.
- **Sarvam Vision** for document OCR.
- **Saaras v4** for speech-to-text and language detection.
- **Sarvam Translate** for English-oriented clinician processing and patient-facing translations.
- **Bulbul v3** for patient voice playback.

> [!WARNING]
> **Not a medical device. Not clinically validated.**
> The bundled rules are illustrative demonstrations marked `ILLUSTRATIVE_DEMO_ONLY_NOT_CLINICALLY_APPROVED`. The system never diagnoses, prescribes, clears patients, or states that a patient does not require care. A qualified clinician must review every output. Do not use real patient data without the required clinical governance, privacy, security, and regulatory reviews.

---

## Web features

### Clinician workstation
- Priority queue with deterministic workflow categories (`EMERGENCY`, `PRIORITY_REVIEW`, `ROUTINE_REVIEW`, `UNASSESSED`).
- Consent gate, encrypted source storage, provenance-linked facts, contradiction review, warnings, timelines, and audit trail.
- Draft clinician note that requires explicit approval.
- Advisory urgency score with an audited clinician override.
- Curated patient communications with staff approval before sending.
- Adaptive follow-up questions with staff approval, send, and dismissal.
- Queue intelligence and conservative wait estimates based on observed reviews.

### Patient experience
- Patient role sees only their own cases, sent messages, and sent/answered follow-up questions. Clinician notes, warnings, discrepancies, raw sources, extracted facts, escalations, and internal processing metadata are excluded from the patient API response and UI.
- A deterministic **patient context card** is built from evidence-linked facts—not the clinician note. It summarizes the main concern, symptoms, medicines, allergies, history, timeline, measurements, and next steps in plain language.
- The patient can choose one of 11 voice-capable Indian languages or upload a short voice sample to detect the language with Saaras v4.
- The whole context card can be translated and read aloud with Bulbul v3.
- Follow-up questions are translated on demand and can be played aloud in the selected language.
- Patients can answer follow-ups by text or by recording/uploading audio. Audio answers preserve the native transcript, retain an English translation for the clinician pipeline, and encrypt the recording at rest.

### Supported voice languages
Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu.

---

## Architecture

```
Browser (static web client)
        │  FastAPI /api
        ▼
CareFlow backend (FastAPI + SQLite + encrypted uploads)
        │
        ├── Sarvam Vision ── document OCR (≤10 pages per job; larger PDFs batched)
        ├── GLM-5.3 ──────── extraction, reconciliation, summary, scoring
        ├── Saaras v4 ────── speech-to-text and language detection
        ├── Sarvam Translate patient/follow-up localization
        └── Bulbul v3 ────── patient voice playback
```

The backend can run locally or on any always-on host. The web client is static and can be served by the backend itself or deployed to Vercel. When the frontend and backend are on different origins, enter the backend URL in the **Backend URL** field in the top navigation bar, or use `?api=https://your-backend.example.com`.

---

## Local setup

### Prerequisites
- Python 3.10+
- FFmpeg on `PATH` (required by `pydub` for audio decoding)
- A Sarvam API key with access to Vision, GLM-5.3, Saaras, Translate, and Bulbul

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

### 2. Configure `.env`

At minimum:

```dotenv
SARVAM_API_KEY=your_key_here
CAREFLOW_LLM_PROVIDER=sarvam
CAREFLOW_ENCRYPTION_KEY=...       # generate with the command below
CAREFLOW_DATA_DIR=./data
CAREFLOW_DB_PATH=./data/careflow.db
```

Generate an encryption key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Never commit `.env`, uploaded documents, the database, or the encryption key.

### 3. Start the backend

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://127.0.0.1:8000/>. Check health:

```bash
curl http://127.0.0.1:8000/api/health
```

The FastAPI process serves both the API and the static client by default.

---

## Deployment

### Backend

Run the FastAPI process on an always-on host with persistent storage for `CAREFLOW_DATA_DIR`. The backend uses SQLite and an encrypted local upload directory; it is not suitable for an ephemeral or read-only serverless filesystem.

Typical process:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Put it behind HTTPS (reverse proxy, tunnel, or platform ingress) and expose it to the browser. The frontend never receives the Sarvam API key.

### Frontend on Vercel

The repository includes `vercel.json` and the static client in `app/static`.

The current demo deployment is:

```text
https://static-theta-self.vercel.app/
```

It was published as a temporary Vercel deployment. Claim it from the Vercel dashboard while signed in if you want to keep it permanently and enable Git-based deployments.

To publish a new deployment from a machine with Vercel credentials:

```bash
npx vercel app/static --yes --prod
```

Or import the Git repository in Vercel and use the included configuration. After deployment, open the site and enter the backend HTTPS URL in the **Backend URL** field, then click **Connect**.

You can also preconfigure a deployment with a query parameter:

```text
https://your-frontend.example.com/?api=https://your-backend.example.com
```

### Frontend on GitHub Pages

A GitHub Actions workflow (`.github/workflows/pages.yml`) publishes the static client to GitHub Pages on every push to `main`. The site is assembled so the page lives at the repository root and its assets live under `static/`, matching the `<base href="./static/">` used by `index.html`.

The deployed demo URL is:

```text
https://rags8i.github.io/sarvam/
```

Enter the backend URL in the web app before using clinical features. GitHub Pages only hosts the static client; it does not run the FastAPI backend or hold any patient data.

### CORS

The backend currently allows cross-origin browser requests. If you deploy publicly, restrict `allow_origins` in `app/main.py` to your frontend origin and enable the `api_key` authentication mode instead of trusted development headers.

---

## Tests

```bash
python -m pytest -q
```

Tests use the deterministic `mock` provider; no live Sarvam key is required. Live voice, OCR, and translation requests are never made by the test suite.

---

## Configuration reference

| Variable | Purpose |
|---|---|
| `SARVAM_API_KEY` | Sarvam API subscription key. Required for live AI features. |
| `CAREFLOW_LLM_PROVIDER` | `sarvam` (default), `ollama` (legacy local), or `mock` (tests only). |
| `CAREFLOW_SARVAM_CHAT_MODEL` | Chat model, default `glm5.3`. |
| `CAREFLOW_SARVAM_BASE_URL` | Default `https://api.sarvam.ai`. |
| `CAREFLOW_OCR_MODEL` | Optional digitise model override. Leave blank to use Sarvam's pipeline default. |
| `CAREFLOW_TTS_MODEL` | Default `bulbul:v3`. |
| `CAREFLOW_TTS_SPEAKER` | Default `shubh`. |
| `CAREFLOW_PATIENT_VOICE_MAX_CHARS` | Maximum context-card characters sent to translation/voice. |
| `CAREFLOW_AUTH_MODE` | `dev` for trusted local headers or `api_key` for keyed access. |
| `CAREFLOW_DATA_DIR` | Persistent data directory. |
| `CAREFLOW_ENCRYPTION_KEY` | Fernet key for data at rest. |

See `.env.example` for the full list, including retention, outbound delivery, and processing settings.

---

## Privacy and limitations

- Sarvam is a **cloud service**. Audio, documents, prompts, and card text leave the local machine when the corresponding features are used. Obtain consent and complete your own data-processing review before using real patient data.
- Sarvam Vision accepts at most 10 pages per job; larger PDFs are rendered and sent in batches.
- Saaras REST requests are chunked below 30 seconds.
- Bulbul supports the 11 languages listed above. Translation may support additional languages, but voice playback is limited to the TTS set.
- Missing information is never treated as a confirmed negative finding.
- Urgent escalation never waits for messages or follow-up answers.
- The repository contains demonstration rules and interfaces only. It is not a substitute for clinical judgment, local protocols, or emergency services.
