import json
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EduAtlas Learner Records API",
    description="AI-powered certificate & student record verification module for online learning platforms",
    version="1.0.0",
)

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./eduatlas.db")
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

VALID_TYPES = {"student", "certificate", "enrollment", "course"}
VALID_STATUSES = {"active", "completed", "expired", "revoked"}


class LearnerRecord(Base):
    __tablename__ = "learner_records"

    id = Column(String, primary_key=True, index=True)
    type = Column(String)
    value = Column(String)
    status = Column(String)
    last_seen = Column(String)
    tags = Column(String)
    record_metadata = Column(String)   # holds source, track, priority, enrichment - everything variable
    integrity_score = Column(Integer, nullable=True)
    ai_summary = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RecordCreate(BaseModel):
    model_config = {"extra": "ignore"}

    id: str = Field(description="Unique stable identifier, e.g. certificate number or student ID")
    type: str = Field(description="One of: student, certificate, enrollment, course")
    value: str = Field(description="Canonical value, e.g. 'CERT-2026-00931', student full name, course code")
    status: str = Field(description="One of: active, completed, expired, revoked")
    last_seen: Optional[datetime] = Field(default=None, description="When the record was last verified/updated")
    tags: List[str] = Field(default=[], description="Free-form labels, e.g. track name, cohort")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Variable fields: source, track, priority, program-specific data")


class RecordQueryFilters(BaseModel):
    type: Optional[str] = Field(None, description="Record type, e.g. 'certificate', 'enrollment'")
    status: Optional[str] = Field(None, description="Record status, e.g. 'expired', 'active'")
    search_term: Optional[str] = Field(None, description="Value or tag to search for, e.g. student name, course code")
    track: Optional[str] = Field(None, description="Program/track classification, e.g. 'AI Engineering', 'Marketing' - checked against tags/metadata")


class RecordVerification(BaseModel):
    integrity_score: int = Field(description="1-10. 10 = highest concern (e.g. expired cert marked active, mismatched dates).")
    priority: str = Field(description="One of: 'critical', 'high', 'medium', 'low' - stored in metadata")
    ai_summary: str = Field(description="One-sentence verification note for an admissions/certification officer.")
    enriched_metadata: Dict[str, str] = Field(description="2 net-new metadata fields inferred from the record, e.g. {'track': 'AI Engineering', 'expiry_status': 'valid'}.")


class IntegrityReport(BaseModel):
    executive_summary: str = Field(description="2-3 sentence summary of overall record integrity across the dataset.")
    critical_findings: List[str] = Field(description="Specific concerns, e.g. expired certificates marked active, duplicate student IDs, revoked certs still referenced.")
    records_by_priority: Dict[str, List[str]] = Field(description="Records grouped by priority: {'critical': [...], 'high': [...], 'medium': [...], 'low': [...]}")
    recommended_actions: List[str] = Field(description="Prioritized list of actions for the certifications/records team.")


# ---------------------------------------------------------------------------
# LLM & chains
# ---------------------------------------------------------------------------

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

verify_prompt = PromptTemplate.from_template("""
You are a records verification analyst for an online learning platform.

Analyze the following learner record and provide a verification assessment:

Record ID: {id}
Type: {type}
Value: {value}
Status: {status}
Tags: {tags}
Metadata: {record_metadata}

Rules:
- integrity_score: 1-10 where 10 = high concern (e.g. expired certificate marked active, inconsistent dates, duplicate-looking ID)
- priority: critical (score 9-10), high (7-8), medium (4-6), low (1-3)
- ai_summary: one sentence a certifications officer would read in a review queue
- enriched_metadata: 2 new fields that add verification context not already in metadata (e.g. inferred 'track', 'expiry_status')

IMPORTANT: Only use information from the record above. Do not invent details.
{error_feedback}
""")

report_prompt = PromptTemplate.from_template("""
You are a records integrity analyst preparing a report for a certifications team lead.

Here is the current learner records inventory:

{inventory}

Analyze the full inventory and produce a structured integrity report.
Focus on: expired certificates still marked active, revoked certificates that are
still referenced elsewhere, stale enrollments, and inconsistent or duplicate-looking records.

IMPORTANT: Only reference records that appear in the inventory above. Do not invent records.
{error_feedback}
""")

query_prompt = PromptTemplate.from_template("""
You are a search assistant for a learner records platform.

Translate this natural language query into structured database filters:

Query: "{request}"

Record types available: student, certificate, enrollment, course
Record statuses available: active, completed, expired, revoked
Tracks available: whatever appears in the data (e.g. AI Engineering, Business, Marketing)

Leave any filter null if not mentioned in the query.
{error_feedback}
""")

verify_chain = verify_prompt | llm.with_structured_output(RecordVerification)
report_chain = report_prompt | llm.with_structured_output(IntegrityReport)
nl_query_chain = query_prompt | llm.with_structured_output(RecordQueryFilters)


def run_feedback_loop(chain, input_data: dict, max_retries: int = 3):
    input_data["error_feedback"] = ""
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"AI failed after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail="AI generation failed after max retries.")
            logger.warning(f"Attempt {attempt} failed. Retrying with error feedback...")
            input_data["error_feedback"] = (
                f"\n--- SYSTEM WARNING ---\n"
                f"Previous output failed validation: {e}\n"
                f"Return strictly formatted JSON only."
            )


def record_to_dict(r: LearnerRecord) -> dict:
    return {
        "id": r.id,
        "type": r.type,
        "value": r.value,
        "status": r.status,
        "last_seen": r.last_seen,
        "tags": json.loads(r.tags) if r.tags else [],
        "metadata": json.loads(r.record_metadata) if r.record_metadata else {},
        "integrity_score": r.integrity_score,
        "ai_summary": r.ai_summary,
    }


def get_priority(r: LearnerRecord) -> str:
    """Priority now lives inside metadata rather than as its own column."""
    meta = json.loads(r.record_metadata) if r.record_metadata else {}
    return meta.get("priority", "unscored")


def validate_record(item: RecordCreate):
    if item.type not in VALID_TYPES:
        raise HTTPException(status_code=422, detail=f"Invalid type '{item.type}'. Must be one of: {VALID_TYPES}")
    if item.status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status '{item.status}'. Must be one of: {VALID_STATUSES}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/records/import", summary="Bulk import learner records (idempotent)")
def bulk_import(records: List[RecordCreate], db: Session = Depends(get_db)):
    """
    Idempotent bulk import. Re-importing an existing record updates last_seen,
    status, tags, and metadata - it does not create a duplicate.
    Malformed records are skipped gracefully and reported in the response.
    """
    added, updated, skipped = 0, 0, []
    seen_in_batch = {}  # id -> LearnerRecord staged in this transaction, not yet committed
    for item in records:
        try:
            validate_record(item)
            existing = seen_in_batch.get(item.id) or db.query(LearnerRecord).filter(LearnerRecord.id == item.id).first()
            if existing:
                existing.last_seen = item.last_seen.isoformat() if item.last_seen else existing.last_seen
                existing.status = item.status
                existing.tags = json.dumps(item.tags)
                existing_meta = json.loads(existing.record_metadata) if existing.record_metadata else {}
                existing_meta.update(item.metadata or {})
                existing.record_metadata = json.dumps(existing_meta)
                updated += 1
            else:
                now = datetime.utcnow().isoformat()
                new_record = LearnerRecord(
                    id=item.id,
                    type=item.type,
                    value=item.value,
                    status=item.status,
                    last_seen=item.last_seen.isoformat() if item.last_seen else now,
                    tags=json.dumps(item.tags),
                    record_metadata=json.dumps(item.metadata or {}),
                    integrity_score=None,
                    ai_summary=None,
                )
                db.add(new_record)
                seen_in_batch[item.id] = new_record
                added += 1
        except HTTPException as e:
            skipped.append({"id": item.id, "reason": e.detail})
        except Exception as e:
            skipped.append({"id": item.id, "reason": str(e)})

    db.commit()
    return {"added": added, "updated": updated, "skipped": skipped}


@app.post("/records/verify/batch", summary="Batch-verify unverified learner records")
def batch_verify(batch_size: int = 5, db: Session = Depends(get_db)):
    """
    Finds records with no integrity_score (unverified) and runs AI verification on them.
    Commits all changes in a single transaction.
    """
    pending = db.query(LearnerRecord).filter(LearnerRecord.integrity_score.is_(None)).limit(batch_size).all()
    if not pending:
        return {"message": "All records are already verified."}

    for record in pending:
        ai_result = run_feedback_loop(verify_chain, {
            "id": record.id,
            "type": record.type,
            "value": record.value,
            "status": record.status,
            "tags": record.tags,
            "record_metadata": record.record_metadata,
        })
        record.integrity_score = ai_result.integrity_score
        record.ai_summary = ai_result.ai_summary
        current_meta = json.loads(record.record_metadata) if record.record_metadata else {}
        current_meta.update(ai_result.enriched_metadata)
        current_meta["priority"] = ai_result.priority
        record.record_metadata = json.dumps(current_meta)

    db.commit()
    return {
        "message": "Batch verification complete.",
        "records_verified": len(pending),
        "processed_ids": [r.id for r in pending],
    }


@app.post("/records/verify/{record_id}", summary="Verify a single learner record with AI analysis")
def verify_record(record_id: str, db: Session = Depends(get_db)):
    """
    Runs the AI verification pipeline on a single record and writes results back to the DB.
    """
    record = db.query(LearnerRecord).filter(LearnerRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    ai_result = run_feedback_loop(verify_chain, {
        "id": record.id,
        "type": record.type,
        "value": record.value,
        "status": record.status,
        "tags": record.tags,
        "record_metadata": record.record_metadata,
    })

    record.integrity_score = ai_result.integrity_score
    record.ai_summary = ai_result.ai_summary
    current_meta = json.loads(record.record_metadata) if record.record_metadata else {}
    current_meta.update(ai_result.enriched_metadata)
    current_meta["priority"] = ai_result.priority
    record.record_metadata = json.dumps(current_meta)

    db.commit()
    return {
        "record_id": record.id,
        "value": record.value,
        "verification": ai_result.model_dump(),
    }


@app.get("/records/report", summary="Generate AI integrity report over the full inventory")
def generate_report(db: Session = Depends(get_db)):
    """
    Generates a structured integrity report grounded in the actual learner records inventory.
    The LLM is explicitly prohibited from referencing records not in the DB.
    """
    records = db.query(LearnerRecord).all()
    if not records:
        raise HTTPException(status_code=404, detail="No records found.")

    inventory = "\n".join(
        f"ID: {r.id} | Type: {r.type} | Value: {r.value} | Status: {r.status} "
        f"| Integrity: {r.integrity_score} | Priority: {get_priority(r)} "
        f"| Tags: {r.tags} | Metadata: {r.record_metadata}"
        for r in records
    )

    ai_report = run_feedback_loop(report_chain, {"inventory": inventory})
    return {
        "report_type": "Learner Records Integrity Report",
        "record_count": len(records),
        "report": ai_report.model_dump(),
    }


@app.get("/records/search", summary="Natural language learner record search")
def nl_search(q: str, db: Session = Depends(get_db)):
    """
    Translates a plain-English query into structured DB filters and returns matching records.
    Example: 'show me expired certificates from the AI Engineering track'
    """
    filters = run_feedback_loop(nl_query_chain, {"request": q})

    query = db.query(LearnerRecord)
    if filters.type:
        query = query.filter(LearnerRecord.type.ilike(f"%{filters.type}%"))
    if filters.status:
        query = query.filter(LearnerRecord.status.ilike(f"%{filters.status}%"))
    if filters.track:
        query = query.filter(
            LearnerRecord.tags.ilike(f"%{filters.track}%") |
            LearnerRecord.record_metadata.ilike(f"%{filters.track}%")
        )
    if filters.search_term:
        query = query.filter(
            LearnerRecord.value.ilike(f"%{filters.search_term}%") |
            LearnerRecord.tags.ilike(f"%{filters.search_term}%")
        )

    results = query.all()
    return {
        "user_query": q,
        "ai_interpreted_filters": filters.model_dump(),
        "total_results": len(results),
        "results": [record_to_dict(r) for r in results],
    }
