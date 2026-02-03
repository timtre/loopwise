"""Loopwise CLI interface."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from loopwise import __version__
from loopwise.config import get_settings, reload_settings
from loopwise.db import (
    create_tables,
    get_issue,
    get_issues,
    get_stats,
    get_suggestion,
    get_suggestions,
    get_trace_with_events,
    get_traces,
    get_unhappy_traces,
    save_issue,
    save_suggestion,
    save_trace,
    update_issue_suggestion,
    update_trace_analysis,
)
from loopwise.models import NormalizedTrace

app = typer.Typer(
    name="loopwise",
    help="Ingest LLM traces, detect issues, and generate improvement suggestions.",
    no_args_is_help=True,
)
console = Console()

# Sub-apps
config_app = typer.Typer(help="Configuration management")
issues_app = typer.Typer(help="View and manage detected issues")
export_app = typer.Typer(help="Export data to files")

app.add_typer(config_app, name="config")
app.add_typer(issues_app, name="issues")
app.add_typer(export_app, name="export")


def parse_duration(duration: str) -> timedelta:
    """Parse duration string like '24h', '7d', '30m' into timedelta."""
    match = re.match(r"^(\d+)([hdm])$", duration.lower())
    if not match:
        raise typer.BadParameter(
            f"Invalid duration format: {duration}. Use format like '24h', '7d', or '30m'"
        )

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "m":
        return timedelta(minutes=value)

    raise typer.BadParameter(f"Unknown duration unit: {unit}")


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"loopwise version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version", callback=version_callback, is_eager=True
    ),
):
    """Loopwise CLI - Analyze LLM traces and generate improvements."""
    # Ensure database tables exist
    create_tables()


# ============================================================================
# Config Commands
# ============================================================================


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
):
    """Set a configuration value."""
    settings = get_settings()
    try:
        settings.set_value(key, value)
        console.print(f"[green]Set {key} successfully[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("show")
def config_show():
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="Loopwise Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description")

    for field_name, field_info in settings.model_fields.items():
        value = getattr(settings, field_name)
        # Mask API keys
        if "api_key" in field_name and value:
            display_value = value[:8] + "..." if len(value) > 8 else "***"
        else:
            display_value = str(value) if value is not None else "[dim]not set[/dim]"

        description = field_info.description or ""
        table.add_row(field_name, display_value, description)

    console.print(table)


# ============================================================================
# Ingest Command
# ============================================================================


@app.command()
def ingest(
    source: str = typer.Option("langsmith", "--source", "-s", help="Trace source platform"),
    since: str = typer.Option("24h", "--since", help="Fetch traces since (e.g., 24h, 7d)"),
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum traces to fetch"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project/workspace name"),
):
    """Ingest traces from an observability platform."""
    settings = get_settings()

    # Parse duration
    try:
        duration = parse_duration(since)
    except typer.BadParameter as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    since_time = datetime.utcnow() - duration

    # Get adapter
    if source.lower() == "langsmith":
        from loopwise.adapters.langsmith import LangSmithAdapter
        try:
            adapter = LangSmithAdapter()
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Unknown source: {source}. Supported: langsmith[/red]")
        raise typer.Exit(1)

    # Fetch and save traces
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Fetching traces from {source}...", total=None)

        try:
            results = adapter.fetch_and_normalize(
                since=since_time,
                limit=limit,
                project=project,
            )
        except Exception as e:
            console.print(f"[red]Error fetching traces: {e}[/red]")
            raise typer.Exit(1)

        progress.update(task, description=f"Saving {len(results)} traces...")

        saved_count = 0
        for trace, events in results:
            try:
                save_trace(trace, events)
                saved_count += 1
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save trace {trace.id}: {e}[/yellow]")

    console.print(f"[green]Successfully ingested {saved_count} traces from {source}[/green]")


# ============================================================================
# Analyze Command
# ============================================================================


@app.command()
def analyze(
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum traces to analyze"),
    reanalyze: bool = typer.Option(False, "--reanalyze", help="Re-analyze already analyzed traces"),
):
    """Analyze traces to detect unhappy sessions."""
    from loopwise.analysis.clustering import create_issues_from_analysis
    from loopwise.analysis.heuristics import analyze_trace

    settings = get_settings()

    # Get traces to analyze
    traces = get_traces(limit=limit, unanalyzed_only=not reanalyze)

    if not traces:
        console.print("[yellow]No traces to analyze[/yellow]")
        return

    console.print(f"Analyzing {len(traces)} traces...")

    analyzed = 0
    unhappy = 0
    issues_created = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing traces...", total=len(traces))

        for trace in traces:
            # Run analysis
            score, results = analyze_trace(trace)

            # Update trace
            update_trace_analysis(trace.id, score, results)
            analyzed += 1

            # Create issues if unhappy
            if score >= settings.unhappiness_threshold:
                unhappy += 1
                # Refresh trace with updated data
                trace.unhappiness_score = score
                trace.heuristic_results = results
                issues = create_issues_from_analysis(trace, results)
                for issue in issues:
                    save_issue(issue)
                    issues_created += 1

            progress.update(task, advance=1)

    # Print summary
    console.print()
    console.print(Panel(
        f"[green]Analyzed:[/green] {analyzed} traces\n"
        f"[yellow]Unhappy:[/yellow] {unhappy} traces ({unhappy/analyzed*100:.1f}%)\n"
        f"[cyan]Issues created:[/cyan] {issues_created}",
        title="Analysis Complete",
    ))


# ============================================================================
# Issues Commands
# ============================================================================


@issues_app.command("list")
def issues_list(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum issues to show"),
):
    """List detected issues."""
    issues = get_issues(limit=limit)

    if not issues:
        console.print("[yellow]No issues found. Run 'loopwise analyze' first.[/yellow]")
        return

    table = Table(title=f"Issues ({len(issues)} shown)")
    table.add_column("ID", style="cyan")
    table.add_column("Heuristic", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Reason")
    table.add_column("Trace ID", style="dim")

    for issue in issues:
        table.add_row(
            issue.id,
            issue.heuristic,
            f"{issue.score:.2f}",
            issue.reason[:50] + "..." if len(issue.reason) > 50 else issue.reason,
            issue.trace_id,
        )

    console.print(table)


@issues_app.command("show")
def issues_show(
    issue_id: str = typer.Argument(..., help="Issue ID"),
):
    """Show details of a specific issue."""
    issue = get_issue(issue_id)

    if not issue:
        console.print(f"[red]Issue not found: {issue_id}[/red]")
        raise typer.Exit(1)

    # Get associated trace
    trace = get_trace_with_events(issue.trace_id)

    console.print(Panel(
        f"[cyan]ID:[/cyan] {issue.id}\n"
        f"[cyan]Heuristic:[/cyan] {issue.heuristic}\n"
        f"[cyan]Score:[/cyan] {issue.score:.2f}\n"
        f"[cyan]Reason:[/cyan] {issue.reason}\n"
        f"[cyan]Evidence:[/cyan]\n{json.dumps(issue.evidence, indent=2, default=str)}\n"
        f"[cyan]Trace ID:[/cyan] {issue.trace_id}\n"
        f"[cyan]Group ID:[/cyan] {issue.group_id or 'ungrouped'}\n"
        f"[cyan]Suggestion:[/cyan] {issue.suggestion_id or 'none'}",
        title=f"Issue: {issue.id}",
    ))

    if trace and trace.events:
        console.print("\n[bold]Associated Trace Events:[/bold]")
        for event in trace.events[:10]:
            error_indicator = "[red][ERROR][/red] " if event.extra_data.get("error") else ""
            console.print(f"  {error_indicator}[{event.type}] {event.extra_data.get('name', 'unnamed')}")
            if event.extra_data.get("error"):
                console.print(f"    Error: {event.extra_data['error'][:100]}")


# ============================================================================
# Suggest Command
# ============================================================================


@app.command()
def suggest(
    issue_id: Optional[str] = typer.Option(None, "--issue", "-i", help="Generate for specific issue"),
    all_issues: bool = typer.Option(False, "--all", "-a", help="Generate for all ungrouped issues"),
):
    """Generate improvement suggestions using LLM analysis."""
    from loopwise.analysis.clustering import get_group_summary, group_issues
    from loopwise.suggestions.generator import SuggestionGenerator

    if not issue_id and not all_issues:
        console.print("[red]Specify --issue <id> or --all[/red]")
        raise typer.Exit(1)

    try:
        generator = SuggestionGenerator()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if issue_id:
        # Generate for single issue
        issue = get_issue(issue_id)
        if not issue:
            console.print(f"[red]Issue not found: {issue_id}[/red]")
            raise typer.Exit(1)

        trace = get_trace_with_events(issue.trace_id)
        traces = [trace] if trace else []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating suggestion...", total=None)
            suggestion = generator.generate_prompt_suggestion([issue], traces)

        if suggestion:
            save_suggestion(suggestion)
            update_issue_suggestion(issue.id, suggestion.id)
            _print_suggestion(suggestion)
        else:
            console.print("[yellow]Could not generate suggestion[/yellow]")

    else:
        # Generate for all issues
        issues = get_issues(without_suggestion=True)

        if not issues:
            console.print("[yellow]No issues without suggestions[/yellow]")
            return

        # Group issues
        groups = group_issues(issues)
        console.print(f"Found {len(issues)} issues in {len(groups)} groups")

        suggestions_created = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating suggestions...", total=len(groups))

            for group_id, group_issues_list in groups.items():
                # Get traces for this group
                trace_ids = list(set(i.trace_id for i in group_issues_list))
                traces = [get_trace_with_events(tid) for tid in trace_ids[:5]]
                traces = [t for t in traces if t]

                suggestion = generator.generate_for_issue_group(
                    group_id, group_issues_list, traces
                )

                if suggestion:
                    save_suggestion(suggestion)
                    suggestions_created += 1

                    # Link issues to suggestion
                    for issue in group_issues_list:
                        update_issue_suggestion(issue.id, suggestion.id)

                progress.update(task, advance=1)

        console.print(f"[green]Created {suggestions_created} suggestions[/green]")


def _print_suggestion(suggestion):
    """Print a suggestion in formatted output."""
    console.print()
    console.print("=" * 60)
    console.print(f"[bold]SUGGESTION:[/bold] {suggestion.title}")
    console.print("=" * 60)
    console.print(f"[cyan]Type:[/cyan]       {suggestion.type}")
    console.print(f"[cyan]Confidence:[/cyan] {int(suggestion.confidence * 100)}%")
    console.print(f"[cyan]Issues:[/cyan]     {', '.join(suggestion.issue_ids[:5])}")

    if suggestion.evidence:
        affected = suggestion.evidence.get("affected_traces", 0)
        console.print(f"[cyan]Affected:[/cyan]   {affected} traces")

    console.print()
    console.print("[bold]PROBLEM:[/bold]")
    console.print(suggestion.description)

    prompt_diff = suggestion.get_prompt_diff()
    if prompt_diff:
        console.print()
        console.print("[bold]RECOMMENDATION:[/bold]")
        console.print(prompt_diff.change_summary)
        console.print()
        console.print("[bold]DIFF:[/bold]")
        console.print(f"  Target: {prompt_diff.target_prompt}")
        console.print()
        if prompt_diff.original and prompt_diff.original != "[Current prompt]":
            console.print("[red]- " + prompt_diff.original[:200] + "[/red]")
        console.print("[green]+ " + prompt_diff.suggested[:500] + "[/green]")

    console.print("=" * 60)


# ============================================================================
# Export Commands
# ============================================================================


@export_app.command("suggestions")
def export_suggestions(
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, markdown)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
):
    """Export suggestions to a file."""
    suggestions = get_suggestions()

    if not suggestions:
        console.print("[yellow]No suggestions to export[/yellow]")
        return

    if format.lower() == "json":
        data = []
        for s in suggestions:
            item = {
                "id": s.id,
                "type": s.type,
                "title": s.title,
                "description": s.description,
                "confidence": s.confidence,
                "issue_ids": s.issue_ids,
                "evidence": s.evidence,
                "prompt_diff": s.prompt_diff,
                "created_at": s.created_at.isoformat(),
            }
            data.append(item)

        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)

    elif format.lower() == "markdown":
        lines = ["# Loopwise Suggestions\n"]

        for s in suggestions:
            lines.append(f"## {s.title}\n")
            lines.append(f"**Type:** {s.type}  ")
            lines.append(f"**Confidence:** {int(s.confidence * 100)}%  ")
            lines.append(f"**ID:** {s.id}\n")
            lines.append(f"### Problem\n{s.description}\n")

            prompt_diff = s.get_prompt_diff()
            if prompt_diff:
                lines.append("### Recommendation\n")
                lines.append(f"{prompt_diff.change_summary}\n")
                lines.append(f"**Target:** {prompt_diff.target_prompt}\n")
                lines.append("```diff")
                if prompt_diff.original and prompt_diff.original != "[Current prompt]":
                    lines.append(f"- {prompt_diff.original}")
                lines.append(f"+ {prompt_diff.suggested}")
                lines.append("```\n")

            lines.append("---\n")

        with open(output, "w") as f:
            f.write("\n".join(lines))

    else:
        console.print(f"[red]Unknown format: {format}. Use 'json' or 'markdown'[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Exported {len(suggestions)} suggestions to {output}[/green]")


# ============================================================================
# Stats Command
# ============================================================================


@app.command()
def stats():
    """Show database statistics."""
    db_stats = get_stats()

    console.print(Panel(
        f"[cyan]Total traces:[/cyan]    {db_stats['total_traces']}\n"
        f"[cyan]Analyzed:[/cyan]        {db_stats['analyzed_traces']}\n"
        f"[yellow]Unhappy:[/yellow]         {db_stats['unhappy_traces']}\n"
        f"[magenta]Total issues:[/magenta]    {db_stats['total_issues']}\n"
        f"[green]Suggestions:[/green]     {db_stats['total_suggestions']}",
        title="Loopwise Statistics",
    ))


if __name__ == "__main__":
    app()
