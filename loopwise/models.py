"""Data models for Loopwise."""

from datetime import datetime
from typing import Any, Literal, Optional
import uuid

from pydantic import BaseModel, Field
from sqlmodel import JSON, Column, Field as SQLField, Relationship, SQLModel


# ============================================================================
# Pydantic Models (for data transfer and validation)
# ============================================================================


class FeedbackData(BaseModel):
    """Feedback data embedded in traces."""

    score: Optional[float] = Field(default=None, description="Score from -1 to 1")
    comment: Optional[str] = Field(default=None, description="User comment")
    source: Literal["user", "auto", "annotation"] = Field(
        default="user", description="Source of feedback"
    )


class HeuristicResult(BaseModel):
    """Result from running a heuristic on a trace."""

    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Heuristic score (0-1)")
    reason: Optional[str] = Field(default=None, description="Why this score was given")
    evidence: Any = Field(default=None, description="Supporting evidence")


class PromptDiff(BaseModel):
    """Suggested changes to a prompt."""

    target_prompt: str = Field(description="Identifier for the target prompt")
    original: str = Field(description="Original prompt text")
    suggested: str = Field(description="Suggested prompt text")
    change_summary: str = Field(description="Summary of what changed")


class ArchitectureChange(BaseModel):
    """Suggested architecture modification."""

    change_type: Literal["remove_tool", "add_tool", "modify_routing", "add_guardrail"] = Field(
        description="Type of architecture change"
    )
    target: str = Field(description="Target component")
    recommendation: str = Field(description="Detailed recommendation")


class KBChange(BaseModel):
    """Suggested knowledge base modification."""

    change_type: Literal["add_document", "update_document", "remove_document", "split_chunk"] = (
        Field(description="Type of KB change")
    )
    target: Optional[str] = Field(default=None, description="Target document/chunk")
    content_suggestion: str = Field(description="Suggested content")
    related_queries: list[str] = Field(default_factory=list, description="Related user queries")


# ============================================================================
# SQLModel Tables (for database persistence)
# ============================================================================


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())[:8]


class TraceEvent(SQLModel, table=True):
    """Individual event within a trace."""

    __tablename__ = "trace_events"

    id: str = SQLField(default_factory=generate_id, primary_key=True)
    trace_id: str = SQLField(foreign_key="traces.id", index=True)
    type: str = SQLField(description="Event type: llm_call, tool_call, retrieval, user_input, agent_output")
    timestamp: datetime = SQLField(default_factory=datetime.utcnow)
    input_text: str = SQLField(default="", description="Event input")
    output_text: str = SQLField(default="", description="Event output")
    duration_ms: int = SQLField(default=0, description="Duration in milliseconds")
    extra_data: dict = SQLField(default_factory=dict, sa_column=Column(JSON))
    parent_id: Optional[str] = SQLField(default=None, description="Parent event ID")

    # Relationship
    trace: Optional["NormalizedTrace"] = Relationship(back_populates="events")


class NormalizedTrace(SQLModel, table=True):
    """Normalized trace from any observability platform."""

    __tablename__ = "traces"

    id: str = SQLField(default_factory=generate_id, primary_key=True)
    source: str = SQLField(description="Source platform: langsmith, langfuse, etc.")
    session_id: str = SQLField(index=True, description="Session/thread ID")
    timestamp: datetime = SQLField(default_factory=datetime.utcnow, index=True)
    extra_data: dict = SQLField(default_factory=dict, sa_column=Column(JSON))
    feedback_score: Optional[float] = SQLField(default=None, description="Feedback score -1 to 1")
    feedback_comment: Optional[str] = SQLField(default=None)
    feedback_source: Optional[str] = SQLField(default=None)

    # Analysis results
    unhappiness_score: Optional[float] = SQLField(default=None, index=True)
    heuristic_results: dict = SQLField(default_factory=dict, sa_column=Column(JSON))
    analyzed_at: Optional[datetime] = SQLField(default=None)

    # Relationships
    events: list[TraceEvent] = Relationship(back_populates="trace")

    @property
    def feedback(self) -> Optional[FeedbackData]:
        """Get feedback as FeedbackData object."""
        if self.feedback_score is None and self.feedback_comment is None:
            return None
        return FeedbackData(
            score=self.feedback_score,
            comment=self.feedback_comment,
            source=self.feedback_source or "user",
        )

    def set_feedback(self, feedback: Optional[FeedbackData]) -> None:
        """Set feedback from FeedbackData object."""
        if feedback is None:
            self.feedback_score = None
            self.feedback_comment = None
            self.feedback_source = None
        else:
            self.feedback_score = feedback.score
            self.feedback_comment = feedback.comment
            self.feedback_source = feedback.source


class Issue(SQLModel, table=True):
    """Detected issue from trace analysis."""

    __tablename__ = "issues"

    id: str = SQLField(default_factory=lambda: f"iss_{generate_id()}", primary_key=True)
    trace_id: str = SQLField(foreign_key="traces.id", index=True)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

    # Issue details
    heuristic: str = SQLField(description="Which heuristic detected this")
    score: float = SQLField(description="Heuristic score")
    reason: str = SQLField(description="Why this was flagged")
    evidence: dict = SQLField(default_factory=dict, sa_column=Column(JSON))

    # Grouping
    group_id: Optional[str] = SQLField(default=None, index=True, description="Issue group ID")

    # Link to suggestion
    suggestion_id: Optional[str] = SQLField(
        default=None, foreign_key="suggestions.id", description="Generated suggestion"
    )


class Suggestion(SQLModel, table=True):
    """Generated improvement suggestion."""

    __tablename__ = "suggestions"

    id: str = SQLField(default_factory=lambda: f"sug_{generate_id()}", primary_key=True)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

    # Suggestion details
    type: str = SQLField(description="Type: prompt, architecture, knowledge_base")
    title: str = SQLField(description="Short title")
    description: str = SQLField(description="Detailed description")
    confidence: float = SQLField(default=0.5, ge=0.0, le=1.0)

    # Evidence
    issue_ids: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))
    evidence: dict = SQLField(default_factory=dict, sa_column=Column(JSON))

    # Type-specific details (stored as JSON)
    prompt_diff: Optional[dict] = SQLField(default=None, sa_column=Column(JSON))
    architecture_change: Optional[dict] = SQLField(default=None, sa_column=Column(JSON))
    kb_change: Optional[dict] = SQLField(default=None, sa_column=Column(JSON))

    def get_prompt_diff(self) -> Optional[PromptDiff]:
        """Get prompt diff as PromptDiff object."""
        if self.prompt_diff is None:
            return None
        return PromptDiff(**self.prompt_diff)

    def set_prompt_diff(self, diff: Optional[PromptDiff]) -> None:
        """Set prompt diff from PromptDiff object."""
        self.prompt_diff = diff.model_dump() if diff else None

    def get_architecture_change(self) -> Optional[ArchitectureChange]:
        """Get architecture change as ArchitectureChange object."""
        if self.architecture_change is None:
            return None
        return ArchitectureChange(**self.architecture_change)

    def set_architecture_change(self, change: Optional[ArchitectureChange]) -> None:
        """Set architecture change from ArchitectureChange object."""
        self.architecture_change = change.model_dump() if change else None

    def get_kb_change(self) -> Optional[KBChange]:
        """Get KB change as KBChange object."""
        if self.kb_change is None:
            return None
        return KBChange(**self.kb_change)

    def set_kb_change(self, change: Optional[KBChange]) -> None:
        """Set KB change from KBChange object."""
        self.kb_change = change.model_dump() if change else None
