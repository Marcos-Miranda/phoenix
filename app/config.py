from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from phoenix.otel import register
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    phoenix_collector_endpoint: str = "http://localhost:4317"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    model: str = "gemma3:12b"


settings = Settings()

tracer_provider = register(endpoint=settings.phoenix_collector_endpoint, batch=False)
AsyncioInstrumentor().instrument(tracer_provider=tracer_provider)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer("app")
