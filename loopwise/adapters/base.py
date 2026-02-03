"""Base adapter protocol for trace ingestion."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from loopwise.models import NormalizedTrace, TraceEvent


class TraceAdapter(ABC):
    """Abstract base class for trace adapters.

    Each adapter is responsible for:
    1. Fetching raw traces from an observability platform
    2. Normalizing them into a common format
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source platform name (e.g., 'langsmith', 'langfuse')."""
        pass

    @abstractmethod
    def fetch_traces(
        self,
        since: datetime,
        limit: int = 100,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw traces from the platform.

        Args:
            since: Fetch traces created after this time
            limit: Maximum number of traces to fetch
            project: Optional project/workspace filter

        Returns:
            List of raw trace dictionaries from the platform
        """
        pass

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> tuple[NormalizedTrace, list[TraceEvent]]:
        """Normalize a raw trace into the common format.

        Args:
            raw: Raw trace data from the platform

        Returns:
            Tuple of (NormalizedTrace, list of TraceEvents)
        """
        pass

    def fetch_and_normalize(
        self,
        since: datetime,
        limit: int = 100,
        project: str | None = None,
    ) -> list[tuple[NormalizedTrace, list[TraceEvent]]]:
        """Fetch and normalize traces in one operation.

        Args:
            since: Fetch traces created after this time
            limit: Maximum number of traces to fetch
            project: Optional project/workspace filter

        Returns:
            List of (NormalizedTrace, events) tuples
        """
        raw_traces = self.fetch_traces(since=since, limit=limit, project=project)
        return [self.normalize(raw) for raw in raw_traces]
