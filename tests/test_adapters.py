"""Tests for adapters module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from loopwise.adapters.langsmith import LangSmithAdapter


class TestLangSmithAdapter:
    """Tests for LangSmith adapter."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        with patch("loopwise.adapters.langsmith.get_settings") as mock:
            settings = MagicMock()
            settings.langsmith_api_key = "test_key"
            settings.langsmith_project = "test_project"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def mock_client(self):
        """Create mock LangSmith client."""
        with patch("loopwise.adapters.langsmith.Client") as mock:
            yield mock.return_value

    def test_source_name(self, mock_settings, mock_client):
        """Test source name property."""
        adapter = LangSmithAdapter()
        assert adapter.source_name == "langsmith"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch("loopwise.adapters.langsmith.get_settings") as mock:
            settings = MagicMock()
            settings.langsmith_api_key = None
            mock.return_value = settings

            with pytest.raises(ValueError) as exc:
                LangSmithAdapter()
            assert "API key required" in str(exc.value)

    def test_normalize_basic_run(self, mock_settings, mock_client):
        """Test normalizing a basic run."""
        adapter = LangSmithAdapter()

        raw_run = {
            "id": "run123",
            "name": "test_run",
            "run_type": "chain",
            "start_time": datetime(2024, 1, 1, 12, 0, 0),
            "end_time": datetime(2024, 1, 1, 12, 0, 5),
            "inputs": {"input": "Hello"},
            "outputs": {"output": "Hi there!"},
            "error": None,
            "extra": {},
            "session_id": "session456",
            "feedback": None,
            "child_runs": [],
        }

        trace, events = adapter.normalize(raw_run)

        assert trace.id == "run123"
        assert trace.source == "langsmith"
        assert trace.session_id == "session456"
        assert len(events) == 1
        assert events[0].type == "agent_output"

    def test_normalize_with_feedback(self, mock_settings, mock_client):
        """Test normalizing a run with feedback."""
        adapter = LangSmithAdapter()

        raw_run = {
            "id": "run123",
            "name": "test_run",
            "run_type": "chain",
            "start_time": datetime(2024, 1, 1, 12, 0, 0),
            "end_time": datetime(2024, 1, 1, 12, 0, 5),
            "inputs": {},
            "outputs": {},
            "error": None,
            "extra": {},
            "session_id": "session456",
            "feedback": {
                "score": -0.5,
                "comment": "Not helpful",
                "feedback_source": "user",
            },
            "child_runs": [],
        }

        trace, events = adapter.normalize(raw_run)

        feedback = trace.feedback
        assert feedback is not None
        assert feedback.score == -0.5
        assert feedback.comment == "Not helpful"

    def test_normalize_with_child_runs(self, mock_settings, mock_client):
        """Test normalizing a run with child runs."""
        adapter = LangSmithAdapter()

        raw_run = {
            "id": "run123",
            "name": "parent_run",
            "run_type": "chain",
            "start_time": datetime(2024, 1, 1, 12, 0, 0),
            "end_time": datetime(2024, 1, 1, 12, 0, 10),
            "inputs": {},
            "outputs": {},
            "error": None,
            "extra": {},
            "session_id": "session456",
            "feedback": None,
            "child_runs": [
                {
                    "id": "child1",
                    "name": "llm_call",
                    "run_type": "llm",
                    "start_time": datetime(2024, 1, 1, 12, 0, 1),
                    "end_time": datetime(2024, 1, 1, 12, 0, 3),
                    "inputs": {"prompt": "Hello"},
                    "outputs": {"text": "Hi"},
                    "error": None,
                    "extra": {},
                },
                {
                    "id": "child2",
                    "name": "tool_call",
                    "run_type": "tool",
                    "start_time": datetime(2024, 1, 1, 12, 0, 4),
                    "end_time": datetime(2024, 1, 1, 12, 0, 5),
                    "inputs": {"query": "search"},
                    "outputs": {"results": []},
                    "error": None,
                    "extra": {},
                },
            ],
        }

        trace, events = adapter.normalize(raw_run)

        assert len(events) == 3  # parent + 2 children
        assert events[0].id == "run123"  # parent
        assert events[1].type == "llm_call"
        assert events[2].type == "tool_call"
        assert events[1].parent_id == "run123"
        assert events[2].parent_id == "run123"

    def test_normalize_with_error(self, mock_settings, mock_client):
        """Test normalizing a run with an error."""
        adapter = LangSmithAdapter()

        raw_run = {
            "id": "run123",
            "name": "failed_run",
            "run_type": "tool",
            "start_time": datetime(2024, 1, 1, 12, 0, 0),
            "end_time": datetime(2024, 1, 1, 12, 0, 1),
            "inputs": {},
            "outputs": {},
            "error": "Connection timeout",
            "extra": {},
            "session_id": "session456",
            "feedback": None,
            "child_runs": [],
        }

        trace, events = adapter.normalize(raw_run)

        assert len(events) == 1
        assert events[0].extra_data.get("error") == "Connection timeout"

    def test_map_run_types(self, mock_settings, mock_client):
        """Test run type mapping to event types."""
        adapter = LangSmithAdapter()

        test_cases = [
            ("llm", "llm_call"),
            ("chat_model", "llm_call"),
            ("tool", "tool_call"),
            ("retriever", "retrieval"),
            ("chain", "agent_output"),
            ("agent", "agent_output"),
            ("unknown", "agent_output"),  # default
        ]

        for run_type, expected in test_cases:
            result = adapter._map_run_type(run_type)
            assert result == expected, f"Failed for {run_type}"
