# EduAtlas — Learner Records & Certificate Verification API

An AI-powered records verification module for online learning platforms. Given a
plain-English question, it returns grounded answers from a real database of
students, enrollments, courses, and certificates — instead of hallucinating them.

Adapted from an original asset-management architecture built for a cybersecurity
attack-surface monitoring platform; the same LLM-grounding, structured-output,
and retry patterns apply directly to certificate/records verification.

---

## Setup

Copy `.env.example` to `.env` and fill in your values:

```
cp .env.example .env
```

| Variable       | Description                                       |
| -------------- | -------------------------------------------------- |
| `DATABASE_URL` | Defaults to `sqlite:///./eduatlas.db` — no install needed. Swap to `postgresql://user:password@localhost:5432/eduatlas` for a production-style setup. |
| `GROQ_API_KEY` | Your Groq API key from console.groq.com             |

## Run Locally

**Prerequisites:** Python 3.11+. No database server needed — SQLite is a local file, created automatically.

```
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API (eduatlas.db is created automatically on first run)
uvicorn main:app --reload

# 4. (Optional) Populate sample data for a demo
python seed_data.py
```

API: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

### Switching to PostgreSQL

For a production-style setup, uncomment the `postgresql://` line in `.env` and comment out the SQLite line, then:

```
pip install psycopg2-binary
createdb eduatlas
```

The app detects the driver from `DATABASE_URL` automatically — no code changes needed.

### Key Libraries

| Library                        | Purpose                             |
| ------------------------------- | ------------------------------------ |
| `fastapi`                       | REST API framework                  |
| `uvicorn`                       | Server that runs FastAPI            |
| `sqlalchemy`                    | ORM — works with SQLite out of the box, Postgres with the extra driver below |
| `pydantic`                      | Request/response validation         |
| `langchain` + `langchain-groq`  | LLM chain framework + Groq provider  |
| `python-dotenv`                 | Loads `.env` variables              |
| `psycopg2-binary` *(optional)*  | Only needed if you switch `DATABASE_URL` to PostgreSQL |

---

## Data Model

Trimmed to 8 columns — everything variable lives in one JSON field instead of
being spread across dedicated columns:

| Column             | Type    | Notes                                                            |
| ------------------- | ------- | ------------------------------------------------------------------ |
| `id`                | string  | Primary key — certificate number, student ID, course code, etc.  |
| `type`              | string  | `student` \| `certificate` \| `enrollment` \| `course`            |
| `value`             | string  | Canonical display value (e.g. student name, cert number)          |
| `status`            | string  | `active` \| `completed` \| `expired` \| `revoked`                  |
| `last_seen`         | string  | Last time the record was imported or verified                     |
| `tags`              | JSON    | Free-form labels (track, cohort, etc.)                             |
| `record_metadata`   | JSON    | Everything variable: source, issued_by, expiry_date, priority, AI-enriched fields |
| `integrity_score`   | int     | 1–10, AI-generated on verification                                 |
| `ai_summary`        | string  | One-line AI verification note                                      |

`priority` and `track` used to be dedicated columns — they're now just keys
inside `record_metadata`, since they're derived/variable rather than core
identity fields. This keeps the schema simple to explain: 5 identity/status
columns, one flexible JSON bucket, two AI-derived fields.

---

## Endpoints

| Method | Path                          | What it does                                                                                                  |
| ------ | ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `POST` | `/records/import`              | Bulk import learner records. Idempotent — re-importing updates `last_seen` and merges metadata, no duplicates. |
| `POST` | `/records/verify/batch`        | Finds all unverified records and runs AI verification on them in one batch.                                     |
| `POST` | `/records/verify/{record_id}`  | Same as batch but for one specific record by ID.                                                                 |
| `GET`  | `/records/report`              | Reads the full records inventory and generates an integrity report. LLM is grounded strictly in DB data.       |
| `GET`  | `/records/search?q=`           | Translate plain English into DB filters. Example: `?q=show me expired certificates from the AI Engineering track` |

---

## Prompts

All three prompts were updated to match the trimmed schema — they no longer
reference `source`, `first_seen`, or `track` as separate fields:

- **`verify_prompt`** — takes `id, type, value, status, tags, record_metadata`. Returns `integrity_score`, `priority` (written into metadata, not a column), `ai_summary`, and `enriched_metadata`.
- **`report_prompt`** — reads the full inventory string (now pulling `priority` out of metadata via `get_priority()`) and produces a structured `IntegrityReport`.
- **`query_prompt`** — translates natural language into `RecordQueryFilters`; the `track` filter now matches against `tags` and `record_metadata` since there's no dedicated `track` column.

## Design Decisions

- **Structured LLM output** — all chains use `with_structured_output()` bound to Pydantic schemas. No free-text parsing.
- **Feedback loop** — on validation failure, the error is injected back into the prompt and retried up to 3 times before returning a 500.
- **Hallucination guard** — `/report` and `/search` are grounded in real DB data. The LLM never invents student or certificate records.
- **Flexible metadata** — variable fields (source, track, priority, issuer, expiry) live in one JSON column instead of being pre-defined as rigid columns, so the schema doesn't need a migration every time a new record type needs a new field.
- **SQLite by default** — zero-install local dev/demo experience; `DATABASE_URL` swaps to PostgreSQL for production with no code changes (the engine auto-detects the driver and sets `check_same_thread` only for SQLite).

## What I'd Do Next

- Add authentication so verification/report endpoints require a certifications-team role
- PII handling: mask student contact info in logs and non-admin responses
- Pagination on search results
- Split into `models.py`, `chains.py`, `routes.py`
- Add a simple rule-based flag for likely-duplicate certificate IDs before the AI pass
