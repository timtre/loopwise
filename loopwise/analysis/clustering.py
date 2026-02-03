"""Issue grouping and clustering."""

import uuid
from typing import Any

from loopwise.models import HeuristicResult, Issue, NormalizedTrace


def generate_group_id() -> str:
    """Generate a unique group ID."""
    return f"grp_{str(uuid.uuid4())[:8]}"


def create_issue(
    trace: NormalizedTrace,
    heuristic_name: str,
    result: HeuristicResult,
) -> Issue:
    """Create an issue from a flagged heuristic result.

    Args:
        trace: The trace that was flagged
        heuristic_name: Name of the heuristic that flagged it
        result: The heuristic result

    Returns:
        New Issue object (not yet saved to database)
    """
    evidence = result.evidence if result.evidence else {}
    if isinstance(evidence, list):
        evidence = {"items": evidence}

    return Issue(
        trace_id=trace.id,
        heuristic=heuristic_name,
        score=result.score,
        reason=result.reason or f"Flagged by {heuristic_name}",
        evidence=evidence,
    )


def create_issues_from_analysis(
    trace: NormalizedTrace,
    heuristic_results: dict[str, dict[str, Any]],
    min_score: float = 0.3,
) -> list[Issue]:
    """Create issues from trace analysis results.

    Args:
        trace: The analyzed trace
        heuristic_results: Results from analyze_trace()
        min_score: Minimum heuristic score to create an issue

    Returns:
        List of Issue objects (not yet saved)
    """
    issues = []
    for name, result_dict in heuristic_results.items():
        score = result_dict.get("score", 0)
        if score >= min_score:
            result = HeuristicResult(
                score=score,
                reason=result_dict.get("reason"),
                evidence=result_dict.get("evidence"),
            )
            issue = create_issue(trace, name, result)
            issues.append(issue)
    return issues


def compute_similarity(issue1: Issue, issue2: Issue) -> float:
    """Compute similarity score between two issues.

    Simple heuristic-based similarity:
    - Same heuristic type: high base similarity
    - Similar evidence (e.g., same tool in loops): bonus

    Args:
        issue1: First issue
        issue2: Second issue

    Returns:
        Similarity score (0-1)
    """
    similarity = 0.0

    # Same heuristic = base similarity of 0.5
    if issue1.heuristic == issue2.heuristic:
        similarity = 0.5

        # Check for specific evidence matches
        ev1 = issue1.evidence or {}
        ev2 = issue2.evidence or {}

        # For tool loops: same tool is very similar
        if issue1.heuristic == "h3_tool_loops":
            if ev1.get("tool") == ev2.get("tool"):
                similarity = 0.9

        # For errors: similar error messages
        if issue1.heuristic == "h2_errors":
            errors1 = ev1.get("errors", [])
            errors2 = ev2.get("errors", [])
            if errors1 and errors2:
                # Check if any error types match
                types1 = {e.get("event_type") for e in errors1 if isinstance(e, dict)}
                types2 = {e.get("event_type") for e in errors2 if isinstance(e, dict)}
                if types1 & types2:
                    similarity = 0.7

    return similarity


def group_issues(issues: list[Issue], similarity_threshold: float = 0.5) -> dict[str, list[Issue]]:
    """Group similar issues together.

    Simple greedy clustering: for each issue, find a group with
    high similarity or create a new group.

    Args:
        issues: List of issues to group
        similarity_threshold: Minimum similarity to join a group

    Returns:
        Dictionary mapping group_id to list of issues
    """
    if not issues:
        return {}

    groups: dict[str, list[Issue]] = {}
    group_representatives: dict[str, Issue] = {}

    for issue in issues:
        best_group = None
        best_similarity = 0.0

        # Find the most similar existing group
        for group_id, representative in group_representatives.items():
            sim = compute_similarity(issue, representative)
            if sim > best_similarity and sim >= similarity_threshold:
                best_similarity = sim
                best_group = group_id

        if best_group:
            # Add to existing group
            issue.group_id = best_group
            groups[best_group].append(issue)
        else:
            # Create new group
            new_group_id = generate_group_id()
            issue.group_id = new_group_id
            groups[new_group_id] = [issue]
            group_representatives[new_group_id] = issue

    return groups


def get_group_summary(issues: list[Issue]) -> dict[str, Any]:
    """Generate a summary for a group of issues.

    Args:
        issues: Issues in the group

    Returns:
        Summary dictionary with common patterns
    """
    if not issues:
        return {}

    # Count heuristics
    heuristic_counts: dict[str, int] = {}
    for issue in issues:
        heuristic_counts[issue.heuristic] = heuristic_counts.get(issue.heuristic, 0) + 1

    # Find common evidence
    common_evidence: dict[str, Any] = {}

    # For tool loops, find common tools
    tool_loop_issues = [i for i in issues if i.heuristic == "h3_tool_loops"]
    if tool_loop_issues:
        tools: dict[str, int] = {}
        for issue in tool_loop_issues:
            tool = issue.evidence.get("tool")
            if tool:
                tools[tool] = tools.get(tool, 0) + 1
        if tools:
            common_evidence["common_tools"] = tools

    # For errors, find common error types
    error_issues = [i for i in issues if i.heuristic == "h2_errors"]
    if error_issues:
        error_types: dict[str, int] = {}
        for issue in error_issues:
            errors = issue.evidence.get("errors", [])
            for err in errors:
                if isinstance(err, dict):
                    err_type = err.get("event_type", "unknown")
                    error_types[err_type] = error_types.get(err_type, 0) + 1
        if error_types:
            common_evidence["common_error_types"] = error_types

    return {
        "issue_count": len(issues),
        "heuristic_counts": heuristic_counts,
        "primary_heuristic": max(heuristic_counts, key=heuristic_counts.get),
        "average_score": sum(i.score for i in issues) / len(issues),
        "common_evidence": common_evidence,
        "trace_ids": [i.trace_id for i in issues],
    }
