"""Service configuration (environment variables prefixed with T2S_, or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="T2S_", env_file=".env", extra="ignore")

    # LLM backend (vLLM's OpenAI-compatible server; Ollama works for CPU-only dev)
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str = "EMPTY"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 256
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    adapters: str = ""  # "spider-v1=/adapters/spider-v1,spider-v2=/adapters/spider-v2"
    default_adapter: str | None = None

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # Data
    data_dir: Path = Path("data")

    # Retrieval (rag / qlora_rag modes)
    rag_num_examples: int = 3
    rag_max_tables: int | None = 6
    rag_value_hints: bool = True

    # Safety + execution
    check_columns: bool = True
    query_timeout_s: float = 5.0
    max_rows: int = 1000
    max_repair_attempts: int = 1

    # Ops
    admin_token: str | None = None
    log_level: str = "INFO"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def db_dirs(self) -> list[Path]:
        """Where SQLite files live: Spider (2024 or 2018 layout) + user databases."""
        raw = self.data_dir / "raw"
        return [
            raw / "spider_data" / "database",
            raw / "spider_data" / "test_database",
            raw / "spider" / "database",
            self.data_dir / "databases",
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
