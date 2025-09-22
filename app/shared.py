import random
from typing import Any

from opentelemetry import trace
from pydantic import BaseModel, Field

tracer = trace.get_tracer(__name__)

ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant but sarcastic when the question is dumb."
GUARDRAIL_FAILURE_MESSAGE = (
    "I'm sorry, I cannot assist with that request because my response was flagged as inappropriate."
)


class GuardrailOutput(BaseModel):
    """Non-toxic output validation."""

    is_compliant: bool = Field(
        description="Indicates if the assistant's message is compliant with non-toxicity standards"
    )
    reasoning: str = Field(description="Reasoning behind the compliance decision")


async def example_traced_operation() -> dict[str, Any]:
    """Example function to demonstrate traditional OpenTelemetry tracing."""
    with tracer.start_as_current_span("example_traced_operation") as span:
        value = random.randint(1, 100)
        span.set_attribute("example.input_value", value)
        span.set_attribute("example.description", "simple increment random number operation")
        result = value + 1
        span.set_attribute("example.result", result)
        return {"input": value, "result": result}
