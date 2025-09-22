from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import settings
from app.shared import ASSISTANT_SYSTEM_PROMPT, GUARDRAIL_FAILURE_MESSAGE, GuardrailOutput, example_traced_operation

tracer = OITracer(trace.get_tracer(__name__), TraceConfig())


openai_client = AsyncOpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    max_retries=1,
    timeout=30,
)


async def answer_question(user_message: ChatCompletionUserMessageParam) -> ChatCompletionAssistantMessageParam:
    system_message = ChatCompletionSystemMessageParam(role="system", content=ASSISTANT_SYSTEM_PROMPT)
    response = await openai_client.chat.completions.create(
        model=settings.model,
        messages=[system_message, user_message],
        temperature=0,
    )
    return response.choices[0].message


async def validate_response(
    user_message: ChatCompletionUserMessageParam, assistant_message: ChatCompletionAssistantMessageParam
) -> GuardrailOutput:
    response = await openai_client.chat.completions.parse(
        model=settings.model,
        messages=[user_message, assistant_message],
        response_format=GuardrailOutput,
        temperature=0,
    )
    parsed = response.choices[0].message.parsed
    if not parsed:
        raise ValueError("Failed to parse the guardrail response")
    return parsed


class MyWorkflow:
    async def start(self, user_question: str) -> ChatCompletionUserMessageParam:
        if user_question.lower() == "exception":
            raise ValueError("Simulated exception for testing purposes")
        _ = await example_traced_operation()
        return ChatCompletionUserMessageParam(role="user", content=user_question)

    @tracer.chain
    async def assistant(self, user_message: ChatCompletionUserMessageParam) -> ChatCompletionAssistantMessageParam:
        return await answer_question(user_message)

    @tracer.chain
    async def guardrail(
        self, user_message: ChatCompletionUserMessageParam, assistant_message: ChatCompletionAssistantMessageParam
    ) -> str:
        guardrail_response = await validate_response(user_message, assistant_message)
        return assistant_message.content if guardrail_response.is_compliant else GUARDRAIL_FAILURE_MESSAGE

    @tracer.chain
    async def run(self, user_question: str) -> str:
        with tracer.start_as_current_span(
            "MyWorkflow.start", openinference_span_kind=OpenInferenceSpanKindValues.CHAIN
        ) as span:
            span.set_input(user_question)
            try:
                user_message = await self.start(user_question)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                return "An error occurred while processing your request."
            span.set_output(user_message)
            span.set_status(Status(StatusCode.OK))
        assistant_message = await self.assistant(user_message)
        return await self.guardrail(user_message, assistant_message)
