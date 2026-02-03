# Loopwise

Ingest LLM traces, detect problematic sessions, and generate improvement suggestions.

Loopwise is an open-source tool that analyzes traces from LLM observability platforms (like LangSmith), automatically detects "unhappy" sessions using heuristics, and generates actionable improvement suggestions for your prompts.

## Features

- **Trace Ingestion**: Import traces from LangSmith (more adapters planned)
- **Unhappy Session Detection**: Automatically flag problematic sessions using heuristics:
  - Negative user feedback
  - Errors in the trace
  - Excessive tool loops
  - High latency
- **LLM-Powered Suggestions**: Generate prompt improvement suggestions using Claude
- **CLI Interface**: Easy-to-use command line tool
- **Export**: Output suggestions as JSON or Markdown

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/loopwise.git
cd loopwise

# Install with Poetry
poetry install
```

## Quick Start

### 1. Configure API Keys

```bash
# Set your LangSmith API key
loopwise config set langsmith_api_key <your-langsmith-key>

# Set your Anthropic API key (for suggestion generation)
loopwise config set anthropic_api_key <your-anthropic-key>

# Optionally set a default project
loopwise config set langsmith_project <your-project-name>
```

### 2. Ingest Traces

```bash
# Ingest traces from the last 24 hours
loopwise ingest --source langsmith --since 24h

# Ingest from a specific project with a limit
loopwise ingest --source langsmith --project my-agent --limit 500
```

### 3. Analyze Traces

```bash
# Run analysis to detect unhappy sessions
loopwise analyze
```

### 4. View Issues

```bash
# List detected issues
loopwise issues list

# View details of a specific issue
loopwise issues show <issue-id>
```

### 5. Generate Suggestions

```bash
# Generate suggestions for all issues
loopwise suggest --all

# Generate suggestion for a specific issue
loopwise suggest --issue <issue-id>
```

### 6. Export Results

```bash
# Export suggestions as JSON
loopwise export suggestions --format json --output suggestions.json

# Export as Markdown
loopwise export suggestions --format markdown --output suggestions.md
```

## CLI Reference

```
Usage: loopwise [OPTIONS] COMMAND [ARGS]...

Commands:
  analyze   Analyze traces to detect unhappy sessions
  config    Configuration management
  export    Export data to files
  ingest    Ingest traces from an observability platform
  issues    View and manage detected issues
  stats     Show database statistics
  suggest   Generate improvement suggestions using LLM analysis
```

### Config Commands

```bash
loopwise config set <key> <value>  # Set a configuration value
loopwise config show               # Show current configuration
```

### Available Configuration Keys

| Key | Description |
|-----|-------------|
| `langsmith_api_key` | LangSmith API key |
| `anthropic_api_key` | Anthropic API key |
| `langsmith_project` | Default LangSmith project |
| `unhappiness_threshold` | Threshold for flagging unhappy sessions (0-1, default: 0.3) |
| `high_latency_threshold_ms` | Latency threshold in ms (default: 30000) |
| `tool_loop_threshold` | Number of tool calls to consider a loop (default: 3) |

## Heuristics

Loopwise uses the following heuristics to detect unhappy sessions:

| Heuristic | Weight | Description |
|-----------|--------|-------------|
| H1: Negative Feedback | 1.0 | User gave negative feedback (score < 0) |
| H2: Errors | 1.0 | Trace contains errors |
| H3: Tool Loops | 0.6 | Same tool called 3+ times (possible loop) |
| H4: High Latency | 0.4 | Total latency exceeds threshold |

Sessions with a weighted score > 0.3 are flagged as unhappy.

## Architecture

```
loopwise/
├── loopwise/
│   ├── cli.py              # Typer CLI
│   ├── config.py           # Settings
│   ├── models.py           # Data models
│   ├── db.py               # Database operations
│   ├── adapters/
│   │   ├── base.py         # Adapter protocol
│   │   └── langsmith.py    # LangSmith implementation
│   ├── analysis/
│   │   ├── heuristics.py   # Unhappiness detection
│   │   └── clustering.py   # Issue grouping
│   └── suggestions/
│       └── generator.py    # LLM suggestion generation
└── tests/
```

## Development

```bash
# Install dependencies (including dev)
poetry install

# Run tests
poetry run pytest tests/ -v

# Run linter
poetry run ruff check loopwise/
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
