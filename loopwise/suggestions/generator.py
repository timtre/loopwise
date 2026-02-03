"""LLM-powered suggestion generation."""

import json
from typing import Any

import anthropic

from loopwise.config import get_settings
from loopwise.models import Issue, NormalizedTrace, PromptDiff, Suggestion

PROMPT_SUGGESTION_SYSTEM = """You are an expert at analyzing LLM agent traces and improving them.
Your task is to analyze issues detected in agent sessions and suggest prompt improvements.

You will receive:
1. A summary of detected issues (grouped by type)
2. Sample trace data showing the problematic interactions

Based on this, suggest specific prompt improvements that would address the issues.

Respond with valid JSON in this exact format:
{
    "title": "Short title for the suggestion (max 80 chars)",
    "description": "Detailed description of the problem and why this change helps",
    "confidence": 0.85,
    "prompt_diff": {
        "target_prompt": "identifier for which prompt to modify (e.g., 'system_prompt')",
        "original": "The problematic part of the current prompt (if identifiable)",
        "suggested": "Your suggested replacement or addition",
        "change_summary": "Brief summary of the change"
    }
}

Guidelines:
- Focus on specific, actionable changes
- If you can't identify the exact original prompt, use "[Current system prompt]" as placeholder
- Confidence should reflect how certain you are this will help (0.5-1.0)
- Be concise but specific in your suggestions
"""


class SuggestionGenerator:
    """Generate improvement suggestions using LLM analysis."""

    def __init__(self, api_key: str | None = None):
        """Initialize the suggestion generator.

        Args:
            api_key: Anthropic API key. If not provided, uses settings.
        """
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key
        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Set via LOOPWISE_ANTHROPIC_API_KEY "
                "or run: loopwise config set anthropic_api_key <key>"
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens

    def generate_prompt_suggestion(
        self,
        issues: list[Issue],
        traces: list[NormalizedTrace],
    ) -> Suggestion | None:
        """Generate a prompt improvement suggestion from issues.

        Args:
            issues: List of related issues
            traces: Traces associated with the issues

        Returns:
            Suggestion object or None if generation fails
        """
        if not issues:
            return None

        # Build context for the LLM
        context = self._build_context(issues, traces)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=PROMPT_SUGGESTION_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": context,
                    }
                ],
            )

            # Parse the response
            response_text = response.content[0].text
            suggestion_data = self._parse_response(response_text)

            if suggestion_data:
                return self._create_suggestion(suggestion_data, issues)

        except Exception as e:
            # Log error but don't crash
            print(f"Error generating suggestion: {e}")

        return None

    def _build_context(
        self,
        issues: list[Issue],
        traces: list[NormalizedTrace],
    ) -> str:
        """Build context string for LLM analysis."""
        lines = ["## Issue Summary\n"]

        # Group issues by heuristic
        by_heuristic: dict[str, list[Issue]] = {}
        for issue in issues:
            if issue.heuristic not in by_heuristic:
                by_heuristic[issue.heuristic] = []
            by_heuristic[issue.heuristic].append(issue)

        for heuristic, heuristic_issues in by_heuristic.items():
            lines.append(f"### {heuristic} ({len(heuristic_issues)} occurrences)")
            for issue in heuristic_issues[:3]:  # Limit samples
                lines.append(f"- Score: {issue.score:.2f}")
                lines.append(f"  Reason: {issue.reason}")
                if issue.evidence:
                    lines.append(f"  Evidence: {json.dumps(issue.evidence, default=str)[:200]}")
            lines.append("")

        # Add trace samples
        lines.append("\n## Sample Trace Data\n")
        for trace in traces[:3]:  # Limit to 3 traces
            lines.append(f"### Trace {trace.id}")
            lines.append(f"- Session: {trace.session_id}")
            lines.append(
                f"- Unhappiness Score: {trace.unhappiness_score:.2f}"
                if trace.unhappiness_score
                else ""
            )

            if trace.feedback:
                lines.append(
                    f"- Feedback: score={trace.feedback.score}, comment={trace.feedback.comment}"
                )

            lines.append("\nEvents:")
            for event in trace.events[:10]:  # Limit events
                lines.append(f"  [{event.type}] {event.extra_data.get('name', 'unnamed')}")
                if event.input_text:
                    lines.append(f"    Input: {event.input_text[:150]}...")
                if event.output_text:
                    lines.append(f"    Output: {event.output_text[:150]}...")
                if event.extra_data.get("error"):
                    lines.append(f"    ERROR: {event.extra_data['error'][:100]}")
            lines.append("")

        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> dict[str, Any] | None:
        """Parse LLM response into suggestion data."""
        try:
            # Try to extract JSON from response
            # Handle cases where response might have markdown code blocks
            text = response_text.strip()
            if text.startswith("```"):
                # Extract from code block
                lines = text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    if line.startswith("```") and in_block:
                        break
                    if in_block:
                        json_lines.append(line)
                text = "\n".join(json_lines)

            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response_text[start:end])
                except json.JSONDecodeError:
                    pass
        return None

    def _create_suggestion(
        self,
        data: dict[str, Any],
        issues: list[Issue],
    ) -> Suggestion:
        """Create a Suggestion object from parsed LLM response."""
        suggestion = Suggestion(
            type="prompt",
            title=data.get("title", "Prompt improvement suggestion"),
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.5),
            issue_ids=[i.id for i in issues],
            evidence={
                "affected_traces": len(set(i.trace_id for i in issues)),
                "issue_count": len(issues),
            },
        )

        # Add prompt diff if present
        if "prompt_diff" in data and data["prompt_diff"]:
            pd = data["prompt_diff"]
            prompt_diff = PromptDiff(
                target_prompt=pd.get("target_prompt", "system_prompt"),
                original=pd.get("original", "[Current prompt]"),
                suggested=pd.get("suggested", ""),
                change_summary=pd.get("change_summary", ""),
            )
            suggestion.set_prompt_diff(prompt_diff)

        return suggestion

    def generate_for_issue_group(
        self,
        group_id: str,
        issues: list[Issue],
        traces: list[NormalizedTrace],
    ) -> Suggestion | None:
        """Generate a suggestion for a group of issues.

        Args:
            group_id: The group ID
            issues: All issues in the group
            traces: Traces associated with the issues

        Returns:
            Suggestion object or None
        """
        suggestion = self.generate_prompt_suggestion(issues, traces)
        if suggestion:
            suggestion.evidence["group_id"] = group_id
        return suggestion
