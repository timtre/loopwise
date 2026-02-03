"""LangSmith adapter for trace ingestion."""

from datetime import datetime
from typing import Any

from langsmith import Client

from loopwise.adapters.base import TraceAdapter
from loopwise.config import get_settings
from loopwise.models import FeedbackData, NormalizedTrace, TraceEvent


class LangSmithAdapter(TraceAdapter):
    """Adapter for ingesting traces from LangSmith."""

    def __init__(self, api_key: str | None = None):
        """Initialize the LangSmith adapter.

        Args:
            api_key: LangSmith API key. If not provided, uses settings.
        """
        settings = get_settings()
        self._api_key = api_key or settings.langsmith_api_key
        if not self._api_key:
            raise ValueError(
                "LangSmith API key required. Set via LOOPWISE_LANGSMITH_API_KEY "
                "or run: loopwise config set langsmith_api_key <key>"
            )
        self._client = Client(api_key=self._api_key)

    @property
    def source_name(self) -> str:
        """Return the source platform name."""
        return "langsmith"

    def fetch_traces(
        self,
        since: datetime,
        limit: int = 100,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw traces (runs) from LangSmith.

        Args:
            since: Fetch runs created after this time
            limit: Maximum number of runs to fetch
            project: Optional project name filter

        Returns:
            List of raw run dictionaries
        """
        settings = get_settings()
        project_name = project or settings.langsmith_project

        # Build filter for top-level runs (traces)
        runs = self._client.list_runs(
            project_name=project_name,
            start_time=since,
            is_root=True,  # Only fetch root runs (traces)
            limit=limit,
        )

        raw_traces = []
        for run in runs:
            # Convert run to dict and fetch child runs
            run_dict = {
                "id": str(run.id),
                "name": run.name,
                "run_type": run.run_type,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "inputs": run.inputs or {},
                "outputs": run.outputs or {},
                "error": run.error,
                "extra": run.extra or {},
                "session_id": str(run.session_id) if run.session_id else str(run.id),
                "feedback": self._fetch_feedback(str(run.id)),
                "child_runs": self._fetch_child_runs(str(run.id), project_name),
            }
            raw_traces.append(run_dict)

        return raw_traces

    def _fetch_child_runs(
        self, parent_run_id: str, project_name: str | None
    ) -> list[dict[str, Any]]:
        """Fetch child runs for a parent run."""
        children = []
        try:
            child_runs = self._client.list_runs(
                project_name=project_name,
                parent_run_id=parent_run_id,
            )
            for child in child_runs:
                children.append({
                    "id": str(child.id),
                    "name": child.name,
                    "run_type": child.run_type,
                    "start_time": child.start_time,
                    "end_time": child.end_time,
                    "inputs": child.inputs or {},
                    "outputs": child.outputs or {},
                    "error": child.error,
                    "extra": child.extra or {},
                })
        except Exception:
            # If we can't fetch children, continue with empty list
            pass
        return children

    def _fetch_feedback(self, run_id: str) -> dict[str, Any] | None:
        """Fetch feedback for a run."""
        try:
            feedbacks = list(self._client.list_feedback(run_ids=[run_id]))
            if feedbacks:
                # Use the most recent feedback
                fb = feedbacks[0]
                return {
                    "score": fb.score,
                    "comment": fb.comment,
                    "feedback_source": getattr(fb, "feedback_source", None),
                }
        except Exception:
            pass
        return None

    def normalize(self, raw: dict[str, Any]) -> tuple[NormalizedTrace, list[TraceEvent]]:
        """Normalize a LangSmith run into common format.

        Args:
            raw: Raw run data from LangSmith

        Returns:
            Tuple of (NormalizedTrace, list of TraceEvents)
        """
        # Create the normalized trace
        trace = NormalizedTrace(
            id=raw["id"],
            source=self.source_name,
            session_id=raw.get("session_id", raw["id"]),
            timestamp=raw["start_time"] if raw.get("start_time") else datetime.utcnow(),
            extra_data={
                "name": raw.get("name"),
                "run_type": raw.get("run_type"),
                "extra": raw.get("extra", {}),
            },
        )

        # Set feedback if present
        if raw.get("feedback"):
            fb = raw["feedback"]
            feedback = FeedbackData(
                score=fb.get("score"),
                comment=fb.get("comment"),
                source=self._map_feedback_source(fb.get("feedback_source")),
            )
            trace.set_feedback(feedback)

        # Create events from the run and its children
        events = []

        # Main run event
        main_event = self._create_event_from_run(raw, parent_id=None)
        events.append(main_event)

        # Child run events
        for child in raw.get("child_runs", []):
            child_event = self._create_event_from_run(child, parent_id=main_event.id)
            events.append(child_event)

        return trace, events

    def _create_event_from_run(
        self, run: dict[str, Any], parent_id: str | None
    ) -> TraceEvent:
        """Create a TraceEvent from a LangSmith run."""
        # Map run_type to our event types
        event_type = self._map_run_type(run.get("run_type", ""))

        # Calculate duration
        duration_ms = 0
        if run.get("start_time") and run.get("end_time"):
            delta = run["end_time"] - run["start_time"]
            duration_ms = int(delta.total_seconds() * 1000)

        # Extract input/output text
        input_text = self._extract_text(run.get("inputs", {}))
        output_text = self._extract_text(run.get("outputs", {}))

        # Build metadata
        metadata = {
            "name": run.get("name"),
            "run_type": run.get("run_type"),
        }

        # Add error if present
        if run.get("error"):
            metadata["error"] = run["error"]

        # Add model info if present (from extra)
        extra = run.get("extra", {})
        if "model" in extra:
            metadata["model"] = extra["model"]
        if "invocation_params" in extra:
            inv = extra["invocation_params"]
            if "model_name" in inv:
                metadata["model"] = inv["model_name"]

        # Add token counts if present
        if "token_usage" in extra:
            metadata["tokens"] = extra["token_usage"]

        return TraceEvent(
            id=run["id"],
            type=event_type,
            timestamp=run.get("start_time") or datetime.utcnow(),
            input_text=input_text,
            output_text=output_text,
            duration_ms=duration_ms,
            extra_data=metadata,
            parent_id=parent_id,
        )

    def _map_run_type(self, run_type: str) -> str:
        """Map LangSmith run_type to our event types."""
        mapping = {
            "llm": "llm_call",
            "chat_model": "llm_call",
            "tool": "tool_call",
            "retriever": "retrieval",
            "chain": "agent_output",
            "agent": "agent_output",
            "prompt": "llm_call",
        }
        return mapping.get(run_type.lower(), "agent_output")

    def _map_feedback_source(self, source: Any) -> str:
        """Map LangSmith feedback source to our format."""
        if source is None:
            return "user"
        source_str = str(source).lower()
        if "auto" in source_str:
            return "auto"
        if "annotation" in source_str or "human" in source_str:
            return "annotation"
        return "user"

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extract text content from input/output dicts."""
        if not data:
            return ""

        # Try common keys
        for key in ["input", "output", "text", "content", "message", "query", "answer"]:
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, list) and value:
                    # Handle message lists
                    if isinstance(value[0], dict) and "content" in value[0]:
                        return value[0]["content"]
                    return str(value[0])
                return str(value)

        # Fall back to string representation of first value
        if data:
            first_value = next(iter(data.values()))
            if isinstance(first_value, str):
                return first_value
            return str(first_value)[:500]  # Truncate long values

        return ""
