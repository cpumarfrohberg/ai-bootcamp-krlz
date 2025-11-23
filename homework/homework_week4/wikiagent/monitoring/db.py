import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from wikiagent.config import DATABASE_URL

logger = logging.getLogger(__name__)

Base = declarative_base()

# Constants
DEFAULT_LOG_LIMIT = 50
DEFAULT_TOTAL_COST = 0.0
DEFAULT_TOTAL_QUERIES = 0
DEFAULT_AVG_COST = 0.0


class LLMLog(Base):
    """Main log table for agent executions."""

    __tablename__ = "llm_logs"

    id = Column(Integer, primary_key=True)
    agent_name = Column(String)
    provider = Column(String)
    model = Column(String)
    user_prompt = Column(Text)
    instructions = Column(Text)
    total_input_tokens = Column(Integer)
    total_output_tokens = Column(Integer)
    assistant_answer = Column(Text)
    raw_json = Column(Text)
    input_cost = Column(Float)
    output_cost = Column(Float)
    total_cost = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    checks = relationship(
        "EvalCheck", back_populates="log", cascade="all, delete-orphan"
    )
    guardrail_events = relationship(
        "GuardrailEvent", back_populates="log", cascade="all, delete-orphan"
    )


class EvalCheck(Base):
    """Evaluation check results."""

    __tablename__ = "eval_checks"

    id = Column(Integer, primary_key=True)
    log_id = Column(
        Integer, ForeignKey("llm_logs.id", ondelete="CASCADE"), nullable=False
    )
    check_name = Column(String, nullable=False)
    passed = Column(Boolean)
    score = Column(Float)
    details = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    log = relationship("LLMLog", back_populates="checks")


class GuardrailEvent(Base):
    """Guardrail monitoring events."""

    __tablename__ = "guardrail_events"

    id = Column(Integer, primary_key=True)
    log_id = Column(
        Integer, ForeignKey("llm_logs.id", ondelete="CASCADE"), nullable=False
    )
    guardrail_name = Column(String, nullable=False)
    triggered = Column(Boolean, nullable=False)
    reason = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    log = relationship("LLMLog", back_populates="guardrail_events")


# Global engine and session factory
_engine: Any | None = None
_SessionLocal: Any | None = None


def _get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            logger.error("DATABASE_URL not set in environment")
            return None
        try:
            _engine = create_engine(DATABASE_URL)
            logger.info("Database engine created")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}", exc_info=True)
            return None
    return _engine


def _get_session_factory():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = _get_engine()
        if not engine:
            return None
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal


@contextmanager
def get_db():
    """Get database session as context manager."""
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set, database unavailable")
        yield None
        return

    SessionLocal = _get_session_factory()
    if not SessionLocal:
        logger.warning("Failed to create session factory, database unavailable")
        yield None
        return

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction failed: {e}", exc_info=True)
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize database - create tables if they don't exist."""
    if not DATABASE_URL:
        logger.error("Cannot initialize database: DATABASE_URL not set")
        return

    try:
        engine = _get_engine()
        if not engine:
            logger.error("Failed to get database engine")
            return
        Base.metadata.create_all(engine)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)


def _handle_db_errors(func):
    """Decorator to handle database errors consistently."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}", exc_info=True)
            return None

    return wrapper


def _insert_record(db: Any, model_class: type, **kwargs) -> int | None:
    """Generic function to insert any model record."""
    if not db:
        logger.warning(
            f"Database session not available, cannot insert {model_class.__name__}"
        )
        return None

    try:
        record = model_class(**kwargs)
        db.add(record)
        db.flush()
        return record.id
    except Exception as e:
        logger.error(f"Failed to insert {model_class.__name__}: {e}", exc_info=True)
        return None


def insert_log(db: Any, **kwargs) -> int | None:
    """Insert log record and return ID. Data should be validated before calling."""
    return _insert_record(db, LLMLog, **kwargs)


def insert_eval_check(db: Any, **kwargs) -> int | None:
    """Insert evaluation check. Data should be validated before calling."""
    return _insert_record(db, EvalCheck, **kwargs)


def insert_guardrail_event(db: Any, **kwargs) -> int | None:
    """Insert guardrail event. Data should be validated before calling."""
    return _insert_record(db, GuardrailEvent, **kwargs)


def _log_to_summary_dict(log: LLMLog) -> dict:
    """Convert LLMLog to summary dictionary."""
    from wikiagent.monitoring.schemas import LogSummaryResponse

    return LogSummaryResponse(
        id=log.id,
        created_at=log.created_at,
        agent_name=log.agent_name,
        model=log.model,
        user_prompt=log.user_prompt,
        total_cost=log.total_cost,
        total_input_tokens=log.total_input_tokens,
        total_output_tokens=log.total_output_tokens,
    ).model_dump()


@_handle_db_errors
def get_recent_logs(limit: int = DEFAULT_LOG_LIMIT) -> list[dict]:
    """Get recent logs for display."""
    with get_db() as db:
        if not db:
            return []

        logs = db.query(LLMLog).order_by(LLMLog.created_at.desc()).limit(limit).all()
        return [_log_to_summary_dict(log) for log in logs]


@_handle_db_errors
def get_cost_stats() -> dict:
    """Get cost statistics."""
    with get_db() as db:
        if not db:
            return {
                "total_cost": DEFAULT_TOTAL_COST,
                "total_queries": DEFAULT_TOTAL_QUERIES,
                "avg_cost": DEFAULT_AVG_COST,
            }

        result = db.query(
            func.sum(LLMLog.total_cost).label("total_cost"),
            func.count(LLMLog.id).label("total_queries"),
        ).first()

        total_cost = float(result.total_cost or DEFAULT_TOTAL_COST)
        total_queries = int(result.total_queries or DEFAULT_TOTAL_QUERIES)
        avg_cost = total_cost / total_queries if total_queries > 0 else DEFAULT_AVG_COST

        return {
            "total_cost": total_cost,
            "total_queries": total_queries,
            "avg_cost": avg_cost,
        }


def _log_to_response_dict(log: LLMLog) -> dict:
    """Convert LLMLog with relationships to response dictionary."""
    from wikiagent.monitoring.schemas import (
        EvalCheckResponse,
        GuardrailEventResponse,
        LogResponse,
    )

    return LogResponse(
        id=log.id,
        created_at=log.created_at,
        agent_name=log.agent_name,
        provider=log.provider,
        model=log.model,
        user_prompt=log.user_prompt,
        instructions=log.instructions,
        assistant_answer=log.assistant_answer,
        total_input_tokens=log.total_input_tokens,
        total_output_tokens=log.total_output_tokens,
        input_cost=log.input_cost,
        output_cost=log.output_cost,
        total_cost=log.total_cost,
        checks=[
            EvalCheckResponse(
                check_name=check.check_name,
                passed=check.passed,
                score=check.score,
                details=check.details,
            )
            for check in log.checks
        ],
        guardrail_events=[
            GuardrailEventResponse(
                guardrail_name=event.guardrail_name,
                triggered=event.triggered,
                reason=event.reason,
            )
            for event in log.guardrail_events
        ],
    ).model_dump()


@_handle_db_errors
def get_log_with_checks(log_id: int) -> dict | None:
    """Get log with all related checks and guardrail events."""
    with get_db() as db:
        if not db:
            return None

        log = db.query(LLMLog).filter(LLMLog.id == log_id).first()
        if not log:
            return None

        return _log_to_response_dict(log)
