# Loopwise: Technical Design Doc

**Version:** 0.1 (Hackathon MVP)
**Date:** January 2026
**Author:** Tim

## Overview

Loopwise is an open-source tool that ingests traces from LLM observability platforms, detects problematic sessions, and generates actionable improvement suggestions for prompts, agent architecture, and knowledge bases.

## Goals (v1)

- Ingest traces from LangSmith (primary), with adapter pattern for future integrations
- Automatically detect "unhappy" sessions using heuristics
- Generate improvement suggestions using LLM analysis
- Output suggestions in structured format (JSON + human-readable)

## Non-Goals (v1)

- Auto-applying suggestions to external systems
- Real-time streaming analysis
- Multi-tenant SaaS infrastructure
- Langfuse/Arize integrations (future)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Loopwise                             │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Ingest    │───▶│   Analyze   │───▶│    Suggest      │  │
│  │   Layer     │    │   Layer     │    │    Layer        │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                  │                   │             │
│         ▼                  ▼                   ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Trace Store │    │Issue Store  │    │Suggestion Store │  │
│  │  (SQLite)   │    │  (SQLite)   │    │    (SQLite)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
         ▲
         │
┌────────┴────────┐
│ Trace Adapters  │
├─────────────────┤
│ - LangSmith     │
│ - Langfuse      │ (future)
│ - OpenTelemetry │ (future)
└─────────────────┘
```

---

## Data Flow

### 1. Ingestion

```
[LangSmith API] → [Adapter] → [Normalized Trace] → [Trace Store]
```

**Trace Adapter Interface:**
```python
class TraceAdapter(Protocol):
    def fetch_traces(self, since: datetime, limit: int) -> list[RawTrace]
    def normalize(self, raw: RawTrace) -> NormalizedTrace
```

**Normalized Trace Schema:**
```python
@dataclass
class NormalizedTrace:
    id: str
    source: str  # "langsmith", "langfuse", etc.
    session_id: str
    timestamp: datetime
    events: list[TraceEvent]
    metadata: dict
    feedback: Optional[FeedbackData]

@dataclass
class TraceEvent:
    id: str
    type: Literal["llm_call", "tool_call", "retrieval", "user_input", "agent_output"]
    timestamp: datetime
    input: str
    output: str
    duration_ms: int
    metadata: dict  # model, tokens, error, etc.
    parent_id: Optional[str]

@dataclass
class FeedbackData:
    score: Optional[float]  # -1 to 1, or None
    comment: Optional[str]
    source: Literal["user", "auto", "annotation"]
```

### 2. Analysis

```
[Trace Store] → [Heuristic Engine] → [Issue Detection] → [Issue Store]
```

Runs periodically or on-demand. Processes traces in batches.

### 3. Suggestion Generation

```
[Issue Store] → [LLM Analyzer] → [Suggestion] → [Suggestion Store]
```

Groups similar issues, generates suggestions using LLM with structured output.

---

## Heuristics for Unhappy Sessions

Each heuristic returns a score (0-1) and optional metadata. Sessions exceeding threshold are flagged.

### H1: Explicit Negative Feedback
```python
def h1_negative_feedback(trace: NormalizedTrace) -> HeuristicResult:
    if trace.feedback and trace.feedback.score is not None:
        if trace.feedback.score < 0:
            return HeuristicResult(score=1.0, reason="User gave negative feedback")
    return HeuristicResult(score=0.0)
```

### H2: Error Detection
```python
def h2_errors(trace: NormalizedTrace) -> HeuristicResult:
    errors = [e for e in trace.events if e.metadata.get("error")]
    if errors:
        return HeuristicResult(
            score=1.0,
            reason=f"Trace contains {len(errors)} error(s)",
            evidence=errors
        )
    return HeuristicResult(score=0.0)
```

### H3: Excessive Tool Loops
```python
def h3_tool_loops(trace: NormalizedTrace) -> HeuristicResult:
    tool_calls = [e for e in trace.events if e.type == "tool_call"]
    tool_counts = Counter(e.metadata.get("tool_name") for e in tool_calls)
    
    for tool, count in tool_counts.items():
        if count >= 3:
            return HeuristicResult(
                score=0.8,
                reason=f"Tool '{tool}' called {count} times (possible loop)",
                evidence={"tool": tool, "count": count}
            )
    return HeuristicResult(score=0.0)
```

### H4: High Latency
```python
def h4_high_latency(trace: NormalizedTrace, threshold_ms: int = 30000) -> HeuristicResult:
    total_duration = sum(e.duration_ms for e in trace.events)
    if total_duration > threshold_ms:
        return HeuristicResult(
            score=0.6,
            reason=f"Total latency {total_duration}ms exceeds threshold",
            evidence={"duration_ms": total_duration}
        )
    return HeuristicResult(score=0.0)
```

### H5: Sentiment Detection (LLM-based)
```python
def h5_user_frustration(trace: NormalizedTrace) -> HeuristicResult:
    user_messages = [e.input for e in trace.events if e.type == "user_input"]
    last_messages = user_messages[-3:]  # Check final messages
    
    # Use LLM to detect frustration signals
    prompt = f"""Analyze these user messages for frustration signals.
    Messages: {last_messages}
    Return JSON: {{"frustrated": bool, "confidence": float, "signals": list[str]}}"""
    
    result = llm_call(prompt)
    if result["frustrated"] and result["confidence"] > 0.7:
        return HeuristicResult(
            score=result["confidence"],
            reason="User frustration detected",
            evidence=result["signals"]
        )
    return HeuristicResult(score=0.0)
```

### H6: Retrieval Relevance (RAG-specific)
```python
def h6_poor_retrieval(trace: NormalizedTrace) -> HeuristicResult:
    retrievals = [e for e in trace.events if e.type == "retrieval"]
    if not retrievals:
        return HeuristicResult(score=0.0)
    
    # Check if retrieved content was actually used in response
    # Use LLM to assess relevance
    for retrieval in retrievals:
        chunks = retrieval.output
        final_response = get_final_response(trace)
        
        relevance = assess_relevance(chunks, final_response)
        if relevance < 0.3:
            return HeuristicResult(
                score=0.7,
                reason="Retrieved content not relevant to response",
                evidence={"chunks": chunks, "relevance_score": relevance}
            )
    return HeuristicResult(score=0.0)
```

### Composite Scoring
```python
def compute_unhappiness_score(trace: NormalizedTrace) -> float:
    weights = {
        "h1_negative_feedback": 1.0,
        "h2_errors": 1.0,
        "h3_tool_loops": 0.6,
        "h4_high_latency": 0.4,
        "h5_user_frustration": 0.8,
        "h6_poor_retrieval": 0.7,
    }
    
    results = run_all_heuristics(trace)
    weighted_sum = sum(results[h].score * w for h, w in weights.items())
    max_possible = sum(weights.values())
    
    return weighted_sum / max_possible
```

**Threshold:** Flag sessions with score > 0.3 as unhappy.

---

## Suggestion Output Format

### Schema
```python
@dataclass
class Suggestion:
    id: str
    type: Literal["prompt", "architecture", "knowledge_base"]
    title: str
    description: str
    confidence: float  # 0-1
    issue_ids: list[str]  # linked issues
    evidence: dict  # supporting data
    
    # Type-specific fields
    prompt_diff: Optional[PromptDiff]
    architecture_change: Optional[ArchitectureChange]
    kb_change: Optional[KBChange]

@dataclass
class PromptDiff:
    target_prompt: str  # identifier
    original: str
    suggested: str
    change_summary: str

@dataclass
class ArchitectureChange:
    change_type: Literal["remove_tool", "add_tool", "modify_routing", "add_guardrail"]
    target: str
    recommendation: str

@dataclass 
class KBChange:
    change_type: Literal["add_document", "update_document", "remove_document", "split_chunk"]
    target: Optional[str]
    content_suggestion: str
    related_queries: list[str]
```

### JSON Output Example
```json
{
  "id": "sug_001",
  "type": "prompt",
  "title": "Add refund policy clarification to system prompt",
  "description": "Users asking about refunds receive inconsistent answers. Adding explicit refund policy to system prompt should reduce confusion.",
  "confidence": 0.85,
  "issue_ids": ["iss_012", "iss_015"],
  "evidence": {
    "affected_traces": 23,
    "sample_queries": ["can I get a refund?", "refund policy", "money back"],
    "failure_pattern": "Agent hallucinates refund terms"
  },
  "prompt_diff": {
    "target_prompt": "customer_support_agent_v2",
    "original": "You are a helpful customer support agent...",
    "suggested": "You are a helpful customer support agent...\n\nREFUND POLICY:\n- Full refund within 30 days\n- Partial refund within 60 days\n- No refund after 60 days",
    "change_summary": "Added explicit refund policy section"
  }
}
```

### CLI Output
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUGGESTION: Add refund policy clarification to system prompt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type:       prompt
Confidence: 85%
Issues:     iss_012, iss_015 (23 affected traces)

PROBLEM:
Users asking about refunds receive inconsistent answers.

RECOMMENDATION:
Add explicit refund policy to system prompt.

DIFF:
  You are a helpful customer support agent...
+ 
+ REFUND POLICY:
+ - Full refund within 30 days
+ - Partial refund within 60 days  
+ - No refund after 60 days
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | LangChain ecosystem, fast prototyping |
| Framework | FastAPI | API endpoints, async support |
| Database | SQLite | Zero setup, good enough for v1 |
| LLM | Claude via Anthropic API | Best for analysis tasks |
| CLI | Typer | Clean CLI interface |
| Tracing SDK | LangSmith SDK | Primary integration |

### Dependencies
```
fastapi
uvicorn
sqlmodel
typer
httpx
anthropic
langsmith
pydantic
rich  # CLI formatting
```

---

## V1 Scope

### In Scope
- [ ] LangSmith trace ingestion (batch, last N hours)
- [ ] Normalized trace storage (SQLite)
- [ ] 4 core heuristics: negative feedback, errors, tool loops, high latency
- [ ] Issue detection and storage
- [ ] LLM-powered suggestion generation (prompts only)
- [ ] CLI for running analysis and viewing results
- [ ] JSON + human-readable output

### Out of Scope (v1)
- Web UI (separate repo/effort)
- Langfuse/Arize adapters
- Real-time ingestion
- Architecture suggestions
- KB suggestions
- Auto-apply to LangSmith Hub
- User accounts / multi-tenancy

---

## File Structure

```
loopwise/
├── loopwise/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI
│   ├── config.py           # Settings
│   ├── models.py           # Pydantic/SQLModel schemas
│   ├── db.py               # Database operations
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py         # Adapter protocol
│   │   └── langsmith.py    # LangSmith implementation
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── heuristics.py   # Unhappiness detection
│   │   └── clustering.py   # Issue grouping
│   └── suggestions/
│       ├── __init__.py
│       └── generator.py    # LLM suggestion generation
├── tests/
├── pyproject.toml
└── README.md
```

---

## CLI Interface

```bash
# Configure
loopwise config set langsmith_api_key <key>
loopwise config set anthropic_api_key <key>

# Ingest traces
loopwise ingest --source langsmith --since 24h --limit 500

# Run analysis
loopwise analyze

# View issues
loopwise issues list
loopwise issues show <issue_id>

# Generate suggestions
loopwise suggest --issue <issue_id>
loopwise suggest --all

# Export
loopwise export suggestions --format json --output suggestions.json
loopwise export suggestions --format markdown --output suggestions.md
```

---

## Open Questions

1. **Clustering granularity:** How similar do issues need to be to group them?
2. **LLM cost management:** Cache analysis results? Limit suggestion generation?
3. **Feedback loop:** How do we track if suggestions were applied and helped?

---

## Success Metrics (Hackathon)

- [ ] Successfully ingest 100+ traces from LangSmith
- [ ] Detect issues in sample dataset with >80% precision
- [ ] Generate 5+ actionable prompt suggestions
- [ ] Demo end-to-end flow in <2 minutes
