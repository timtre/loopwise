"""Tests for heuristics module."""

from datetime import datetime

from loopwise.analysis.heuristics import (
    HEURISTICS,
    compute_score_from_results,
    compute_unhappiness_score,
    h1_negative_feedback,
    h2_errors,
    h3_tool_loops,
    h4_high_latency,
    run_all_heuristics,
)
from loopwise.models import FeedbackData, HeuristicResult, NormalizedTrace, TraceEvent


def create_trace(events: list[TraceEvent] | None = None, **kwargs) -> NormalizedTrace:
    """Helper to create a trace with events."""
    trace = NormalizedTrace(
        id="test_trace",
        source="test",
        session_id="test_session",
        timestamp=datetime.utcnow(),
        **kwargs,
    )
    trace.events = events or []
    return trace


class TestH1NegativeFeedback:
    """Tests for negative feedback heuristic."""

    def test_no_feedback(self):
        """Test trace with no feedback."""
        trace = create_trace()
        result = h1_negative_feedback(trace)
        assert result.score == 0.0

    def test_positive_feedback(self):
        """Test trace with positive feedback."""
        trace = create_trace()
        trace.set_feedback(FeedbackData(score=0.8, source="user"))
        result = h1_negative_feedback(trace)
        assert result.score == 0.0

    def test_negative_feedback(self):
        """Test trace with negative feedback."""
        trace = create_trace()
        trace.set_feedback(FeedbackData(score=-0.5, comment="Bad", source="user"))
        result = h1_negative_feedback(trace)
        assert result.score == 1.0
        assert "negative feedback" in result.reason.lower()


class TestH2Errors:
    """Tests for error detection heuristic."""

    def test_no_errors(self):
        """Test trace with no errors."""
        events = [
            TraceEvent(
                id="e1",
                trace_id="test",
                type="llm_call",
                extra_data={"model": "gpt-4"},
            )
        ]
        trace = create_trace(events=events)
        result = h2_errors(trace)
        assert result.score == 0.0

    def test_with_errors(self):
        """Test trace with errors."""
        events = [
            TraceEvent(
                id="e1",
                trace_id="test",
                type="tool_call",
                extra_data={"error": "Connection timeout"},
            ),
            TraceEvent(
                id="e2",
                trace_id="test",
                type="llm_call",
                extra_data={"error": "Rate limit exceeded"},
            ),
        ]
        trace = create_trace(events=events)
        result = h2_errors(trace)
        assert result.score == 1.0
        assert "2 error" in result.reason


class TestH3ToolLoops:
    """Tests for tool loop detection heuristic."""

    def test_no_tool_calls(self):
        """Test trace with no tool calls."""
        events = [TraceEvent(id="e1", trace_id="test", type="llm_call", extra_data={})]
        trace = create_trace(events=events)
        result = h3_tool_loops(trace)
        assert result.score == 0.0

    def test_normal_tool_usage(self):
        """Test trace with normal tool usage (no loops)."""
        events = [
            TraceEvent(id="e1", trace_id="test", type="tool_call", extra_data={"name": "search"}),
            TraceEvent(
                id="e2", trace_id="test", type="tool_call", extra_data={"name": "calculator"}
            ),
        ]
        trace = create_trace(events=events)
        result = h3_tool_loops(trace)
        assert result.score == 0.0

    def test_tool_loop_detected(self):
        """Test trace with tool loop (same tool called 3+ times)."""
        events = [
            TraceEvent(id=f"e{i}", trace_id="test", type="tool_call", extra_data={"name": "search"})
            for i in range(5)
        ]
        trace = create_trace(events=events)
        result = h3_tool_loops(trace)
        assert result.score == 0.8
        assert "search" in result.reason
        assert "5 times" in result.reason


class TestH4HighLatency:
    """Tests for high latency heuristic."""

    def test_normal_latency(self):
        """Test trace with normal latency."""
        events = [
            TraceEvent(id="e1", trace_id="test", type="llm_call", duration_ms=1000),
            TraceEvent(id="e2", trace_id="test", type="tool_call", duration_ms=500),
        ]
        trace = create_trace(events=events)
        result = h4_high_latency(trace)
        assert result.score == 0.0

    def test_high_latency(self):
        """Test trace with high latency."""
        events = [
            TraceEvent(id="e1", trace_id="test", type="llm_call", duration_ms=20000),
            TraceEvent(id="e2", trace_id="test", type="llm_call", duration_ms=15000),
        ]
        trace = create_trace(events=events)
        result = h4_high_latency(trace)
        assert result.score == 0.6
        assert "35000ms" in result.reason


class TestRunAllHeuristics:
    """Tests for running all heuristics."""

    def test_run_all(self):
        """Test running all heuristics on a trace."""
        trace = create_trace()
        results = run_all_heuristics(trace)

        assert len(results) == len(HEURISTICS)
        for name in HEURISTICS:
            assert name in results
            assert isinstance(results[name], HeuristicResult)


class TestComputeUnhappinessScore:
    """Tests for computing unhappiness scores."""

    def test_happy_trace(self):
        """Test score for a happy trace."""
        trace = create_trace()
        score = compute_unhappiness_score(trace)
        assert score == 0.0

    def test_unhappy_trace(self):
        """Test score for an unhappy trace."""
        trace = create_trace()
        trace.set_feedback(FeedbackData(score=-1.0, source="user"))

        events = [
            TraceEvent(
                id="e1",
                trace_id="test",
                type="tool_call",
                extra_data={"error": "Failed"},
            )
        ]
        trace.events = events

        score = compute_unhappiness_score(trace)
        # Should have high score due to negative feedback and error
        assert score > 0.5

    def test_score_from_results(self):
        """Test computing score from heuristic results."""
        results = {
            "h1_negative_feedback": HeuristicResult(score=1.0),
            "h2_errors": HeuristicResult(score=0.0),
            "h3_tool_loops": HeuristicResult(score=0.0),
            "h4_high_latency": HeuristicResult(score=0.0),
        }
        score = compute_score_from_results(results)
        # h1 has weight 1.0, total weight is 3.0 (1.0 + 1.0 + 0.6 + 0.4)
        expected = 1.0 / 3.0
        assert abs(score - expected) < 0.01
