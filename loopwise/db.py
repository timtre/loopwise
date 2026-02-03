"""Database operations for Loopwise."""

from datetime import datetime
from typing import Optional

from sqlmodel import Session, SQLModel, create_engine, select

from loopwise.config import get_settings
from loopwise.models import Issue, NormalizedTrace, Suggestion, TraceEvent


# Global engine instance
_engine = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, echo=False)
    return _engine


def create_tables():
    """Create all database tables."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    return Session(get_engine())


# ============================================================================
# Trace Operations
# ============================================================================


def save_trace(trace: NormalizedTrace, events: list[TraceEvent]) -> NormalizedTrace:
    """Save a trace and its events to the database."""
    with get_session() as session:
        session.add(trace)
        for event in events:
            event.trace_id = trace.id
            session.add(event)
        session.commit()
        session.refresh(trace)
        return trace


def get_trace(trace_id: str) -> Optional[NormalizedTrace]:
    """Get a trace by ID."""
    with get_session() as session:
        return session.get(NormalizedTrace, trace_id)


def get_trace_with_events(trace_id: str) -> Optional[NormalizedTrace]:
    """Get a trace with all its events loaded."""
    with get_session() as session:
        statement = select(NormalizedTrace).where(NormalizedTrace.id == trace_id)
        trace = session.exec(statement).first()
        if trace:
            # Eagerly load events
            _ = trace.events
        return trace


def get_traces(
    since: Optional[datetime] = None,
    limit: int = 100,
    unanalyzed_only: bool = False,
) -> list[NormalizedTrace]:
    """Get traces with optional filtering."""
    with get_session() as session:
        statement = select(NormalizedTrace)

        if since:
            statement = statement.where(NormalizedTrace.timestamp >= since)

        if unanalyzed_only:
            statement = statement.where(NormalizedTrace.analyzed_at.is_(None))

        statement = statement.order_by(NormalizedTrace.timestamp.desc()).limit(limit)
        traces = session.exec(statement).all()

        # Eagerly load events for each trace
        for trace in traces:
            _ = trace.events

        return list(traces)


def update_trace_analysis(
    trace_id: str,
    unhappiness_score: float,
    heuristic_results: dict,
) -> Optional[NormalizedTrace]:
    """Update trace with analysis results."""
    with get_session() as session:
        trace = session.get(NormalizedTrace, trace_id)
        if trace:
            trace.unhappiness_score = unhappiness_score
            trace.heuristic_results = heuristic_results
            trace.analyzed_at = datetime.utcnow()
            session.add(trace)
            session.commit()
            session.refresh(trace)
        return trace


def get_unhappy_traces(threshold: float = 0.3, limit: int = 100) -> list[NormalizedTrace]:
    """Get traces flagged as unhappy."""
    with get_session() as session:
        statement = (
            select(NormalizedTrace)
            .where(NormalizedTrace.unhappiness_score.isnot(None))
            .where(NormalizedTrace.unhappiness_score >= threshold)
            .order_by(NormalizedTrace.unhappiness_score.desc())
            .limit(limit)
        )
        traces = session.exec(statement).all()
        for trace in traces:
            _ = trace.events
        return list(traces)


def trace_exists(source: str, external_id: str) -> bool:
    """Check if a trace with the given source and external ID already exists."""
    with get_session() as session:
        statement = select(NormalizedTrace).where(
            NormalizedTrace.source == source,
            NormalizedTrace.id == external_id,
        )
        return session.exec(statement).first() is not None


# ============================================================================
# Issue Operations
# ============================================================================


def save_issue(issue: Issue) -> Issue:
    """Save an issue to the database."""
    with get_session() as session:
        session.add(issue)
        session.commit()
        session.refresh(issue)
        return issue


def get_issue(issue_id: str) -> Optional[Issue]:
    """Get an issue by ID."""
    with get_session() as session:
        return session.get(Issue, issue_id)


def get_issues(
    limit: int = 100,
    ungrouped_only: bool = False,
    without_suggestion: bool = False,
) -> list[Issue]:
    """Get issues with optional filtering."""
    with get_session() as session:
        statement = select(Issue)

        if ungrouped_only:
            statement = statement.where(Issue.group_id.is_(None))

        if without_suggestion:
            statement = statement.where(Issue.suggestion_id.is_(None))

        statement = statement.order_by(Issue.created_at.desc()).limit(limit)
        return list(session.exec(statement).all())


def get_issues_by_trace(trace_id: str) -> list[Issue]:
    """Get all issues for a trace."""
    with get_session() as session:
        statement = select(Issue).where(Issue.trace_id == trace_id)
        return list(session.exec(statement).all())


def get_issues_by_group(group_id: str) -> list[Issue]:
    """Get all issues in a group."""
    with get_session() as session:
        statement = select(Issue).where(Issue.group_id == group_id)
        return list(session.exec(statement).all())


def update_issue_group(issue_id: str, group_id: str) -> Optional[Issue]:
    """Update issue group assignment."""
    with get_session() as session:
        issue = session.get(Issue, issue_id)
        if issue:
            issue.group_id = group_id
            session.add(issue)
            session.commit()
            session.refresh(issue)
        return issue


def update_issue_suggestion(issue_id: str, suggestion_id: str) -> Optional[Issue]:
    """Link an issue to a suggestion."""
    with get_session() as session:
        issue = session.get(Issue, issue_id)
        if issue:
            issue.suggestion_id = suggestion_id
            session.add(issue)
            session.commit()
            session.refresh(issue)
        return issue


# ============================================================================
# Suggestion Operations
# ============================================================================


def save_suggestion(suggestion: Suggestion) -> Suggestion:
    """Save a suggestion to the database."""
    with get_session() as session:
        session.add(suggestion)
        session.commit()
        session.refresh(suggestion)
        return suggestion


def get_suggestion(suggestion_id: str) -> Optional[Suggestion]:
    """Get a suggestion by ID."""
    with get_session() as session:
        return session.get(Suggestion, suggestion_id)


def get_suggestions(limit: int = 100) -> list[Suggestion]:
    """Get all suggestions."""
    with get_session() as session:
        statement = (
            select(Suggestion).order_by(Suggestion.created_at.desc()).limit(limit)
        )
        return list(session.exec(statement).all())


def get_suggestions_by_type(suggestion_type: str, limit: int = 100) -> list[Suggestion]:
    """Get suggestions filtered by type."""
    with get_session() as session:
        statement = (
            select(Suggestion)
            .where(Suggestion.type == suggestion_type)
            .order_by(Suggestion.created_at.desc())
            .limit(limit)
        )
        return list(session.exec(statement).all())


# ============================================================================
# Statistics
# ============================================================================


def get_stats() -> dict:
    """Get database statistics."""
    with get_session() as session:
        trace_count = session.exec(select(NormalizedTrace)).all()
        analyzed_count = session.exec(
            select(NormalizedTrace).where(NormalizedTrace.analyzed_at.isnot(None))
        ).all()
        unhappy_count = session.exec(
            select(NormalizedTrace).where(NormalizedTrace.unhappiness_score >= 0.3)
        ).all()
        issue_count = session.exec(select(Issue)).all()
        suggestion_count = session.exec(select(Suggestion)).all()

        return {
            "total_traces": len(trace_count),
            "analyzed_traces": len(analyzed_count),
            "unhappy_traces": len(unhappy_count),
            "total_issues": len(issue_count),
            "total_suggestions": len(suggestion_count),
        }
