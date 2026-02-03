"""Tests for data models."""

from datetime import datetime

import pytest

from loopwise.models import (
    FeedbackData,
    HeuristicResult,
    Issue,
    NormalizedTrace,
    PromptDiff,
    Suggestion,
    TraceEvent,
)


class TestFeedbackData:
    """Tests for FeedbackData model."""

    def test_create_feedback(self):
        """Test creating feedback data."""
        feedback = FeedbackData(score=0.8, comment="Great!", source="user")
        assert feedback.score == 0.8
        assert feedback.comment == "Great!"
        assert feedback.source == "user"

    def test_feedback_defaults(self):
        """Test feedback default values."""
        feedback = FeedbackData()
        assert feedback.score is None
        assert feedback.comment is None
        assert feedback.source == "user"


class TestHeuristicResult:
    """Tests for HeuristicResult model."""

    def test_create_result(self):
        """Test creating heuristic result."""
        result = HeuristicResult(score=0.8, reason="Test reason", evidence={"key": "value"})
        assert result.score == 0.8
        assert result.reason == "Test reason"
        assert result.evidence == {"key": "value"}

    def test_score_bounds(self):
        """Test score must be between 0 and 1."""
        with pytest.raises(ValueError):
            HeuristicResult(score=1.5)

        with pytest.raises(ValueError):
            HeuristicResult(score=-0.1)


class TestNormalizedTrace:
    """Tests for NormalizedTrace model."""

    def test_create_trace(self):
        """Test creating a normalized trace."""
        trace = NormalizedTrace(
            id="test123",
            source="langsmith",
            session_id="session456",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            extra_data={"name": "test"},
        )
        assert trace.id == "test123"
        assert trace.source == "langsmith"
        assert trace.session_id == "session456"

    def test_feedback_property(self):
        """Test feedback getter/setter."""
        trace = NormalizedTrace(
            id="test123",
            source="langsmith",
            session_id="session456",
        )

        # Initially no feedback
        assert trace.feedback is None

        # Set feedback
        feedback = FeedbackData(score=-0.5, comment="Bad", source="user")
        trace.set_feedback(feedback)

        assert trace.feedback_score == -0.5
        assert trace.feedback_comment == "Bad"

        # Get feedback back
        retrieved = trace.feedback
        assert retrieved is not None
        assert retrieved.score == -0.5
        assert retrieved.comment == "Bad"


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_create_event(self):
        """Test creating a trace event."""
        event = TraceEvent(
            id="event123",
            trace_id="trace456",
            type="llm_call",
            input_text="Hello",
            output_text="Hi there!",
            duration_ms=500,
            extra_data={"model": "gpt-4"},
        )
        assert event.id == "event123"
        assert event.type == "llm_call"
        assert event.duration_ms == 500


class TestIssue:
    """Tests for Issue model."""

    def test_create_issue(self):
        """Test creating an issue."""
        issue = Issue(
            trace_id="trace123",
            heuristic="h1_negative_feedback",
            score=1.0,
            reason="User gave negative feedback",
            evidence={"feedback_score": -1.0},
        )
        assert issue.trace_id == "trace123"
        assert issue.heuristic == "h1_negative_feedback"
        assert issue.score == 1.0

    def test_issue_id_prefix(self):
        """Test issue IDs have correct prefix."""
        issue = Issue(
            trace_id="trace123",
            heuristic="h2_errors",
            score=0.8,
            reason="Error detected",
        )
        assert issue.id.startswith("iss_")


class TestSuggestion:
    """Tests for Suggestion model."""

    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = Suggestion(
            type="prompt",
            title="Add clarification",
            description="Add clarification to system prompt",
            confidence=0.85,
            issue_ids=["iss_123", "iss_456"],
        )
        assert suggestion.type == "prompt"
        assert suggestion.confidence == 0.85
        assert len(suggestion.issue_ids) == 2

    def test_prompt_diff_property(self):
        """Test prompt diff getter/setter."""
        suggestion = Suggestion(
            type="prompt",
            title="Test",
            description="Test",
        )

        # Set prompt diff
        diff = PromptDiff(
            target_prompt="system_prompt",
            original="You are helpful",
            suggested="You are helpful. Always be concise.",
            change_summary="Added conciseness instruction",
        )
        suggestion.set_prompt_diff(diff)

        # Get it back
        retrieved = suggestion.get_prompt_diff()
        assert retrieved is not None
        assert retrieved.target_prompt == "system_prompt"
        assert "concise" in retrieved.suggested

    def test_suggestion_id_prefix(self):
        """Test suggestion IDs have correct prefix."""
        suggestion = Suggestion(
            type="prompt",
            title="Test",
            description="Test",
        )
        assert suggestion.id.startswith("sug_")
