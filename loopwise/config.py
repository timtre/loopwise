"""Application configuration and settings."""

import json
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_config_dir() -> Path:
    """Get the loopwise config directory, creating it if needed."""
    config_dir = Path.home() / ".loopwise"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get the path to the config file."""
    return get_config_dir() / "config.json"


def get_db_path() -> Path:
    """Get the path to the SQLite database."""
    return get_config_dir() / "loopwise.db"


class Settings(BaseSettings):
    """Application settings loaded from environment and config file."""

    model_config = SettingsConfigDict(
        env_prefix="LOOPWISE_",
        env_file=".env",
        extra="ignore",
    )

    # API Keys
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # Database
    database_url: str = Field(
        default=f"sqlite:///{get_db_path()}",
        description="SQLite database URL",
    )

    # Heuristic thresholds
    unhappiness_threshold: float = Field(
        default=0.3,
        description="Threshold for flagging sessions as unhappy (0-1)",
    )
    high_latency_threshold_ms: int = Field(
        default=30000,
        description="Latency threshold in milliseconds",
    )
    tool_loop_threshold: int = Field(
        default=3,
        description="Number of repeated tool calls to consider a loop",
    )

    # LLM settings
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model to use for analysis",
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM responses",
    )

    # LangSmith settings
    langsmith_project: Optional[str] = Field(
        default=None,
        description="Default LangSmith project to ingest from",
    )

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from config file and environment."""
        config_file = get_config_file()
        file_settings = {}

        if config_file.exists():
            with open(config_file) as f:
                file_settings = json.load(f)

        return cls(**file_settings)

    def save(self) -> None:
        """Save current settings to config file."""
        config_file = get_config_file()

        # Only save non-default, non-None values
        data = {}
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)
            if value is not None and value != field_info.default:
                data[field_name] = value

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

    def set_value(self, key: str, value: str) -> None:
        """Set a configuration value and save."""
        if key not in self.model_fields:
            raise ValueError(f"Unknown configuration key: {key}")

        # Convert value to appropriate type
        field_info = self.model_fields[key]
        if field_info.annotation is int:
            value = int(value)
        elif field_info.annotation is float:
            value = float(value)

        setattr(self, key, value)
        self.save()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from disk."""
    global _settings
    _settings = Settings.load()
    return _settings
