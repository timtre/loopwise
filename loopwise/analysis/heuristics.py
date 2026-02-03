"""Heuristics for detecting unhappy sessions."""

from collections import Counter
from typing import Any

from loopwise.config import get_settings
from loopwise.models import HeuristicResult, NormalizedTrace, TraceEvent


def h1_negative_feedback(trace: NormalizedTrace) -> HeuristicResult:
    """H1: Check for explicit negative feedback.

    Returns score of 1.0 if user gave negative feedback (score < 0).
    """
    feedback = trace.feedback
    if feedback and feedback.score is not None:
        if feedback.score < 0:
            return HeuristicResult(
                score=1.0,
                reason="User gave negative feedback",
                evidence={
                    "feedback_score": feedback.score,
                    "comment": feedback.comment,
                },
            )
    return HeuristicResult(score=0.0)


def h2_errors(trace: NormalizedTrace) -> HeuristicResult:
    """H2: Detect errors in trace events.

    Returns score of 1.0 if any events contain errors.
    """
    errors = []
    for event in trace.events:
        error = event.extra_data.get("error")
        if error:
            errors.append({
                "event_id": event.id,
                "event_type": event.type,
                "error": str(error)[:200],  # Truncate long errors
            })

    if errors:
        return HeuristicResult(
            score=1.0,
            reason=f"Trace contains {len(errors)} error(s)",
            evidence={"errors": errors},
        )
    return HeuristicResult(score=0.0)


def h3_tool_loops(trace: NormalizedTrace) -> HeuristicResult:
    """H3: Detect excessive tool call repetition.

    Returns score of 0.8 if any tool is called >= threshold times.
    """
    settings = get_settings()
    threshold = settings.tool_loop_threshold

    tool_calls = [e for e in trace.events if e.type == "tool_call"]
    tool_counts: Counter[str] = Counter()

    for event in tool_calls:
        tool_name = event.extra_data.get("name") or event.extra_data.get("tool_name", "unknown")
        tool_counts[tool_name] += 1

    for tool, count in tool_counts.most_common():
        if count >= threshold:
            return HeuristicResult(
                score=0.8,
                reason=f"Tool '{tool}' called {count} times (possible loop)",
                evidence={
                    "tool": tool,
                    "count": count,
                    "threshold": threshold,
                },
            )

    return HeuristicResult(score=0.0)


def h4_high_latency(trace: NormalizedTrace) -> HeuristicResult:
    """H4: Flag sessions exceeding latency threshold.

    Returns score of 0.6 if total duration exceeds threshold.
    """
    settings = get_settings()
    threshold_ms = settings.high_latency_threshold_ms

    total_duration = sum(e.duration_ms for e in trace.events)

    if total_duration > threshold_ms:
        return HeuristicResult(
            score=0.6,
            reason=f"Total latency {total_duration}ms exceeds threshold ({threshold_ms}ms)",
            evidence={
                "duration_ms": total_duration,
                "threshold_ms": threshold_ms,
            },
        )

    return HeuristicResult(score=0.0)


# Registry of all heuristics with their weights
HEURISTICS: dict[str, tuple[callable, float]] = {
    "h1_negative_feedback": (h1_negative_feedback, 1.0),
    "h2_errors": (h2_errors, 1.0),
    "h3_tool_loops": (h3_tool_loops, 0.6),
    "h4_high_latency": (h4_high_latency, 0.4),
}


def run_all_heuristics(trace: NormalizedTrace) -> dict[str, HeuristicResult]:
    """Run all heuristics on a trace.

    Args:
        trace: The trace to analyze

    Returns:
        Dictionary mapping heuristic name to result
    """
    results = {}
    for name, (heuristic_fn, _) in HEURISTICS.items():
        try:
            results[name] = heuristic_fn(trace)
        except Exception as e:
            # If a heuristic fails, record the error but continue
            results[name] = HeuristicResult(
                score=0.0,
                reason=f"Heuristic failed: {str(e)}",
            )
    return results


def compute_unhappiness_score(trace: NormalizedTrace) -> float:
    """Compute weighted unhappiness score for a trace.

    Args:
        trace: The trace to analyze

    Returns:
        Weighted unhappiness score (0-1)
    """
    results = run_all_heuristics(trace)
    return compute_score_from_results(results)


def compute_score_from_results(results: dict[str, HeuristicResult]) -> float:
    """Compute weighted score from heuristic results.

    Args:
        results: Dictionary of heuristic results

    Returns:
        Weighted unhappiness score (0-1)
    """
    weighted_sum = 0.0
    max_possible = 0.0

    for name, (_, weight) in HEURISTICS.items():
        if name in results:
            weighted_sum += results[name].score * weight
        max_possible += weight

    if max_possible == 0:
        return 0.0

    return weighted_sum / max_possible


def analyze_trace(trace: NormalizedTrace) -> tuple[float, dict[str, Any]]:
    """Analyze a trace and return unhappiness score with details.

    Args:
        trace: The trace to analyze

    Returns:
        Tuple of (unhappiness_score, heuristic_results_dict)
    """
    results = run_all_heuristics(trace)
    score = compute_score_from_results(results)

    # Convert results to serializable dict
    results_dict = {
        name: {
            "score": result.score,
            "reason": result.reason,
            "evidence": result.evidence,
        }
        for name, result in results.items()
    }

    return score, results_dict


def is_unhappy(trace: NormalizedTrace, threshold: float | None = None) -> bool:
    """Check if a trace should be flagged as unhappy.

    Args:
        trace: The trace to check
        threshold: Custom threshold (defaults to settings value)

    Returns:
        True if trace is unhappy
    """
    if threshold is None:
        settings = get_settings()
        threshold = settings.unhappiness_threshold

    score = compute_unhappiness_score(trace)
    return score >= threshold
