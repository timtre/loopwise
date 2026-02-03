# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies
poetry install

# Run CLI
poetry run loopwise --help

# Run tests
poetry run pytest tests/ -v

# Run a single test file
poetry run pytest tests/test_heuristics.py -v

# Lint
poetry run ruff check loopwise/

# Lint and auto-fix
poetry run ruff check --fix loopwise/
```

## Architecture

Loopwise is a CLI tool that ingests LLM traces from observability platforms, detects "unhappy" sessions via heuristics, and generates improvement suggestions using Claude.

### Data Flow

```
Trace Sources → Adapters → Database → Heuristics Analysis → Issue Detection → LLM Suggestions
```

### Key Modules

- **cli.py** - Typer-based CLI entry point (`loopwise.cli:app`)
- **config.py** - Settings management with Pydantic-Settings (stored at `~/.loopwise/config.json`)
- **models.py** - SQLModel data models: `NormalizedTrace`, `TraceEvent`, `Issue`, `Suggestion`
- **db.py** - SQLite database operations (stored at `~/.loopwise/loopwise.db`)
- **adapters/** - Pluggable trace source adapters (currently LangSmith; extensible via `base.py` protocol)
- **analysis/heuristics.py** - Weighted heuristics for unhappiness detection (H1-H4)
- **analysis/clustering.py** - Groups similar issues together
- **suggestions/generator.py** - Anthropic Claude integration for generating prompt improvements

### Heuristics System

Four heuristics with weighted scoring (threshold 0.3):
- H1: Negative Feedback (weight 1.0)
- H2: Errors (weight 1.0)
- H3: Tool Loops (weight 0.6) - same tool called 3+ times
- H4: High Latency (weight 0.4)

### Configuration

Settings via environment variables (`LOOPWISE_*` prefix) or `~/.loopwise/config.json`:
- `langsmith_api_key`, `anthropic_api_key` - Required API credentials
- `unhappiness_threshold` (default 0.3), `high_latency_threshold_ms` (default 30000), `tool_loop_threshold` (default 3)
- `llm_model` (default "claude-sonnet-4-20250514")

## Code Style

- Python 3.11+
- Line length: 100 characters
- Ruff linter with rules: E, F, I, N, W
- Full type hints using Pydantic models
- Pytest with asyncio_mode = "auto"
