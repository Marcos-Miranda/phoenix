from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from app.config import settings


def set_tracer_provider() -> None:
    resource = Resource(attributes={ResourceAttributes.PROJECT_NAME: "app"})
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    trace.set_tracer_provider(tracer_provider)
    # instrumentations
    AsyncioInstrumentor().instrument(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
