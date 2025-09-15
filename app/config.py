from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    tracer_method: Literal["otel", "phoenix"] = "otel"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    model: str = "gemma3:12b"


settings = Settings()
