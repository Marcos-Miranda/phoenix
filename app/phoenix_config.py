from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from phoenix.otel import register

from app.config import settings


def set_tracer_provider() -> None:
    tracer_provider = register(endpoint=settings.phoenix_collector_endpoint, protocol="http/protobuf", batch=False)
    AsyncioInstrumentor().instrument(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
